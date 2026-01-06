import triton
import torch
import triton.language as tl
import numpy as np

# =============================================================================
# CACHED VERSION: Supports cache (conv_state) for step-by-step execution
# =============================================================================
# Similar to causal_conv1d_fn, this version:
# - Takes conv_state as input for temporal history cache
# - Uses cache_indices to map sequences to cache lines
# - Uses has_initial_state to determine whether to read from cache
# - Updates cache with final state after processing

@triton.jit
def depthwise_strided_conv2d_cached_kernel(
    # Pointers
    x_ptr,              # input: (cu_time_in, frequency, channels) - packed sequences
    w_ptr,              # kernel: (3, 3, channels)
    bias_ptr,           # bias: (channels,)
    out_ptr,            # output: (cu_time_out, frequency//2, channels) - packed sequences
    # Cache pointers (like conv_states in causal_conv1d)
    conv_state_ptr,     # cache: (num_cache_lines, freq_len, channels) - stores last odd time values
    cache_indices_ptr,  # (batch,) int32 - maps sequence to cache line index
    has_initial_state_ptr,  # (batch,) bool - whether to use cached state
    # Sequence mapping (computed on CPU, passed to kernel)
    batch_ptr,          # (num_programs,) maps program_id -> sequence index
    time_chunk_offset_ptr,  # (num_programs,) maps program_id -> chunk index within sequence
    # Sequence boundaries
    query_start_loc_in_ptr,   # (batch+1,) cumulative input time positions
    query_start_loc_out_ptr,  # (batch+1,) cumulative output time positions
    # Dimensions
    FREQ_LEN: tl.constexpr,
    TOTAL_CHANNELS: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides for conv_state
    stride_state_seq: tl.constexpr,
    stride_state_freq: tl.constexpr,
    stride_state_ch: tl.constexpr,
    # Block sizes
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_TIME: tl.constexpr,  # Number of output time steps per chunk
):
    """
    Varlen depthwise strided 2D conv with cache support for step-by-step execution.
    
    Grid: (num_programs, freq_out, channel_blocks)
    - program_id(0): sequence/chunk program (from batch_ptr/time_chunk_offset_ptr)
    - program_id(1): output frequency index  
    - program_id(2): channel block index
    
    Cache layout: (num_cache_lines, freq_len, channels)
    - Stores the values from the last processed odd time step
    - When chunk_offset == 0 and has_initial_state == True, reads from cache
    - After processing, updates cache with final odd time values
    """
    # ==========================================================================
    # Step 1: Map program_id to (sequence_idx, chunk_offset)
    # ==========================================================================
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(time_chunk_offset_ptr + tl.program_id(0))
    
    f_out = tl.program_id(1)  # Output frequency index
    pid_ch = tl.program_id(2)  # Channel block index
    
    idx_ch = pid_ch * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    mask_ch = idx_ch < TOTAL_CHANNELS
    
    # ==========================================================================
    # Step 2: Get cache index for this sequence
    # ==========================================================================
    cache_idx = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)
    
    # ==========================================================================
    # Step 3: Locate this sequence in the packed input/output
    # ==========================================================================
    seq_start_in = tl.load(query_start_loc_in_ptr + idx_seq)
    seq_start_out = tl.load(query_start_loc_out_ptr + idx_seq)
    seq_end_out = tl.load(query_start_loc_out_ptr + idx_seq + 1)
    seq_time_out = seq_end_out - seq_start_out
    
    # ==========================================================================
    # Step 4: Compute chunk boundaries within this sequence
    # ==========================================================================
    t_out_start = chunk_offset * BLOCK_TIME
    t_out_end = tl.minimum(t_out_start + BLOCK_TIME, seq_time_out)
    chunk_len = t_out_end - t_out_start
    
    if chunk_len <= 0:
        return
    
    # ==========================================================================
    # Step 5: Setup frequency indices and strides
    # ==========================================================================
    f_hi = f_out * 2 + 1
    f_mid = f_out * 2
    f_lo = f_out * 2 - 1
    valid_f_lo = f_lo >= 0
    
    # Strides for packed layout (time, freq, channels)
    x_stride_t = FREQ_LEN * TOTAL_CHANNELS
    x_stride_f = TOTAL_CHANNELS
    out_stride_t = (FREQ_LEN // 2) * TOTAL_CHANNELS
    out_stride_f = TOTAL_CHANNELS
    w_stride_t = 3 * TOTAL_CHANNELS
    w_stride_f = TOTAL_CHANNELS
    
    # ==========================================================================
    # Step 6: Load kernel weights
    # ==========================================================================
    w00 = tl.load(w_ptr + 0*w_stride_t + 0*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w01 = tl.load(w_ptr + 0*w_stride_t + 1*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w02 = tl.load(w_ptr + 0*w_stride_t + 2*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w10 = tl.load(w_ptr + 1*w_stride_t + 0*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w11 = tl.load(w_ptr + 1*w_stride_t + 1*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w12 = tl.load(w_ptr + 1*w_stride_t + 2*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w20 = tl.load(w_ptr + 2*w_stride_t + 0*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w21 = tl.load(w_ptr + 2*w_stride_t + 1*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    w22 = tl.load(w_ptr + 2*w_stride_t + 2*w_stride_f + idx_ch, mask=mask_ch, other=0.0)
    bias = tl.load(bias_ptr + idx_ch, mask=mask_ch, other=0.0)
    
    # ==========================================================================
    # Step 7: Initialize temporal cache (like causal_conv1d)
    # ==========================================================================
    if chunk_offset == 0:
        # First chunk: check if we should load from cache
        load_init_state = tl.load(has_initial_state_ptr + idx_seq).to(tl.int1)
        
        if load_init_state:
            # Load from conv_state cache
            # Cache layout: (num_cache_lines, freq_len, channels)
            cache_base = conv_state_ptr + cache_idx * stride_state_seq
            
            cache_f_hi = tl.load(
                cache_base + f_hi * stride_state_freq + idx_ch * stride_state_ch,
                mask=mask_ch, other=0.0
            )
            cache_f_mid = tl.load(
                cache_base + f_mid * stride_state_freq + idx_ch * stride_state_ch,
                mask=mask_ch, other=0.0
            )
            cache_f_lo = tl.load(
                cache_base + f_lo * stride_state_freq + idx_ch * stride_state_ch,
                mask=mask_ch & valid_f_lo, other=0.0
            )
        else:
            # No cached state, initialize to zeros
            cache_f_hi = tl.zeros((BLOCK_CHANNELS,), dtype=x_ptr.dtype.element_ty)
            cache_f_mid = tl.zeros((BLOCK_CHANNELS,), dtype=x_ptr.dtype.element_ty)
            cache_f_lo = tl.zeros((BLOCK_CHANNELS,), dtype=x_ptr.dtype.element_ty)
    else:
        # Subsequent chunk: load from previous chunk's last odd time in x
        t_prev_odd = (t_out_start - 1) * 2 + 1
        t_abs_prev = seq_start_in + t_prev_odd
        
        cache_f_hi = tl.load(x_ptr + t_abs_prev * x_stride_t + f_hi * x_stride_f + idx_ch,
                             mask=mask_ch, other=0.0)
        cache_f_mid = tl.load(x_ptr + t_abs_prev * x_stride_t + f_mid * x_stride_f + idx_ch,
                              mask=mask_ch, other=0.0)
        cache_f_lo = tl.load(x_ptr + t_abs_prev * x_stride_t + f_lo * x_stride_f + idx_ch,
                             mask=mask_ch & valid_f_lo, other=0.0)
    
    # ==========================================================================
    # Step 8: Main convolution loop over time chunk
    # ==========================================================================
    # Track the last odd time values for cache update
    last_x_odd_hi = cache_f_hi
    last_x_odd_mid = cache_f_mid
    last_x_odd_lo = cache_f_lo
    
    for t_local in range(BLOCK_TIME):
        t_out_seq = t_out_start + t_local
        valid_t = t_out_seq < seq_time_out
        
        t_odd_seq = t_out_seq * 2 + 1
        t_even_seq = t_out_seq * 2
        t_odd_abs = seq_start_in + t_odd_seq
        t_even_abs = seq_start_in + t_even_seq
        
        # Load input values
        x_odd_hi = tl.load(x_ptr + t_odd_abs * x_stride_t + f_hi * x_stride_f + idx_ch,
                           mask=mask_ch & valid_t, other=0.0)
        x_odd_mid = tl.load(x_ptr + t_odd_abs * x_stride_t + f_mid * x_stride_f + idx_ch,
                            mask=mask_ch & valid_t, other=0.0)
        x_odd_lo = tl.load(x_ptr + t_odd_abs * x_stride_t + f_lo * x_stride_f + idx_ch,
                           mask=mask_ch & valid_f_lo & valid_t, other=0.0)
        x_even_hi = tl.load(x_ptr + t_even_abs * x_stride_t + f_hi * x_stride_f + idx_ch,
                            mask=mask_ch & valid_t, other=0.0)
        x_even_mid = tl.load(x_ptr + t_even_abs * x_stride_t + f_mid * x_stride_f + idx_ch,
                             mask=mask_ch & valid_t, other=0.0)
        x_even_lo = tl.load(x_ptr + t_even_abs * x_stride_t + f_lo * x_stride_f + idx_ch,
                            mask=mask_ch & valid_f_lo & valid_t, other=0.0)
        
        # Use cached values for t_prev
        x_prev_hi = cache_f_hi
        x_prev_mid = cache_f_mid
        x_prev_lo = cache_f_lo
        
        # Compute 2D conv
        out = (w00 * x_odd_hi  + w01 * x_odd_mid  + w02 * x_odd_lo +
               w10 * x_even_hi + w11 * x_even_mid + w12 * x_even_lo +
               w20 * x_prev_hi + w21 * x_prev_mid + w22 * x_prev_lo + bias)
        
        # Store result
        t_out_abs = seq_start_out + t_out_seq
        tl.store(out_ptr + t_out_abs * out_stride_t + f_out * out_stride_f + idx_ch,
                 out, mask=mask_ch & valid_t)
        
        # Update caches for next iteration
        cache_f_hi = x_odd_hi
        cache_f_mid = x_odd_mid
        cache_f_lo = x_odd_lo
        
        # Track last valid values for final cache update
        if valid_t:
            last_x_odd_hi = x_odd_hi
            last_x_odd_mid = x_odd_mid
            last_x_odd_lo = x_odd_lo
    
    # ==========================================================================
    # Step 9: Update conv_state cache with final state
    # Only the last chunk for each sequence should update the cache
    # We check if this is the last chunk by seeing if t_out_end == seq_time_out
    # ==========================================================================
    is_last_chunk = (t_out_end >= seq_time_out)
    if is_last_chunk:
        cache_base = conv_state_ptr + cache_idx * stride_state_seq
        
        tl.store(
            cache_base + f_hi * stride_state_freq + idx_ch * stride_state_ch,
            last_x_odd_hi, mask=mask_ch
        )
        tl.store(
            cache_base + f_mid * stride_state_freq + idx_ch * stride_state_ch,
            last_x_odd_mid, mask=mask_ch
        )
        tl.store(
            cache_base + f_lo * stride_state_freq + idx_ch * stride_state_ch,
            last_x_odd_lo, mask=mask_ch & valid_f_lo
        )


def depthwise_strided_conv2d_cached(
    x: torch.Tensor,                    # (cu_time_in, freq, channels) - packed sequences
    w: torch.Tensor,                    # (3, 3, channels)
    bias: torch.Tensor,                 # (channels,)
    out: torch.Tensor,                  # (cu_time_out, freq//2, channels) - pre-allocated output
    conv_state: torch.Tensor,           # (num_cache_lines, freq, channels) - cache
    query_start_loc: torch.Tensor,      # (batch+1,) cumulative input time positions
    cache_indices: torch.Tensor,        # (batch,) maps sequence to cache line
    has_initial_state: torch.Tensor,    # (batch,) bool - whether to use cached state
    block_ch: int = 256,
    block_t: int = 64,
    metadata=None,                      # Optional: provides pre-computed batch_ptr, etc.
) -> torch.Tensor:
    """
    Varlen depthwise strided 2D convolution with cache support.
    
    Similar to causal_conv1d_fn, this supports step-by-step execution where:
    - conv_state stores temporal history from previous tokens
    - cache_indices maps each sequence to its cache line
    - has_initial_state indicates whether to read from cache
    
    If metadata is provided, uses pre-computed batch_ptr/time_chunk_offset_ptr from it
    (no CPU blocking). Otherwise computes on the fly (CPU blocking - not for CUDA graphs).
    
    Args:
        x: Packed input tensor (cu_time_in, freq, channels)
        w: Kernel weights (3, 3, channels)
        bias: Bias (channels,)
        out: Pre-allocated output tensor (cu_time_out, freq//2, channels)
        conv_state: Cache tensor (num_cache_lines, freq, channels)
                   Stores the last processed odd time step values.
                   Updated in-place after processing.
        query_start_loc: Cumulative sequence boundaries (batch+1,)
        cache_indices: Maps each sequence to cache line (batch,)
        has_initial_state: Whether to use cached state (batch,) bool
        block_ch: Channel block size
        block_t: Time block size
        metadata: Optional metadata with pre-computed batch_ptr, time_chunk_offset_ptr,
                  query_start_loc_out, num_programs (for CUDA graph compatibility)
    
    Returns:
        Output tensor (cu_time_out, freq//2, channels)
    """
    cu_time_in, freq_len, channels = x.shape
    freq_out = freq_len // 2
    
    assert freq_len % 2 == 0, "Frequency dimension must be even"
    assert conv_state.shape[1] >= freq_len, "conv_state frequency dimension must be >= freq_len"
    assert conv_state.shape[2] == channels, "conv_state channels dimension must be == channels"
    
    # Get batch_ptr, time_chunk_offset_ptr, query_start_loc_out from metadata or compute
    if metadata is not None:
        # Use pre-computed values from metadata (no CPU blocking)
        batch_ptr = metadata.batch_ptr
        time_chunk_offset_ptr = metadata.time_chunk_offset_ptr
        query_start_loc_out = metadata.query_start_loc_out
        num_programs = metadata.num_programs
    else:
        # Compute on the fly (CPU blocking - not safe for CUDA graphs)
        query_start_loc_cpu = query_start_loc.cpu()
        seqlens_in = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seqlens_out = seqlens_in // 2
        
        batch_size = len(seqlens_in)
        
        query_start_loc_out_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        query_start_loc_out_cpu[1:] = torch.cumsum(seqlens_out, dim=0)
        
        # Build program mapping
        batch_list = []
        chunk_offset_list = []
        
        for seq_idx, seq_out_len in enumerate(seqlens_out.numpy()):
            num_chunks = int(np.ceil(seq_out_len / block_t))
            if num_chunks == 0:
                num_chunks = 1
            batch_list.extend([seq_idx] * num_chunks)
            chunk_offset_list.extend(range(num_chunks))
        
        num_programs = len(batch_list)
        
        batch_ptr = torch.tensor(batch_list, dtype=torch.int32, device=x.device)
        time_chunk_offset_ptr = torch.tensor(chunk_offset_list, dtype=torch.int32, device=x.device)
        query_start_loc_out = query_start_loc_out_cpu.to(x.device)
    
    # Get strides
    stride_state_seq, stride_state_freq, stride_state_ch = conv_state.stride()
    
    # Launch kernel
    grid = (num_programs, freq_out, triton.cdiv(channels, block_ch))

    depthwise_strided_conv2d_cached_kernel[grid](
        x, w, bias, out,
        conv_state, cache_indices, has_initial_state,
        batch_ptr, time_chunk_offset_ptr,
        query_start_loc, query_start_loc_out,
        FREQ_LEN=freq_len,
        TOTAL_CHANNELS=channels,
        num_cache_lines=conv_state.shape[0],
        stride_state_seq=stride_state_seq,
        stride_state_freq=stride_state_freq,
        stride_state_ch=stride_state_ch,
        BLOCK_CHANNELS=block_ch,
        BLOCK_TIME=block_t,
    )
    
    return out