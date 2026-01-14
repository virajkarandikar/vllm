import triton
import torch
import triton.language as tl
import numpy as np


# =============================================================================
# CACHED VERSION: Supports cache (stft_state) for streaming execution
# =============================================================================
# Simplified version that ALWAYS uses cache:
# - Cache is always read (zeros for first call / prefill)
# - Cache is always updated with last samples after processing
# - No has_initial_state needed - just initialize cache to zeros before first use
#
# Cache size = N_FFT - HOP (the overlap between consecutive STFT frames)
# Virtual input = [cache | new_samples], so for streaming:
#   - Input of HOP samples + cache of (N_FFT - HOP) samples = N_FFT total = 1 frame
#
# For proper streaming, input lengths should be divisible by hop_length.
# This ensures clean frame boundaries: num_frames = seq_len // hop_length
#
# This kernel uses FastConformerConvMetadata-compatible inputs:
# - query_start_loc: cumulative positions in base units
# - batch_ptr / token_chunk_offset_ptr: program ID to (seq, chunk) mapping
# - TIME_FACTOR: multiply query_start_loc to get sample positions
# - OUTPUT_DIVISOR: divide sample positions to get frame positions (= hop_length)

PAD_SLOT_ID = -1


@triton.jit
def stft_cached_kernel(
    X_ptr,              # Input: (total_samples,) - packed sequences
    W_real_ptr,         # Real basis (Freq, N_FFT)
    W_imag_ptr,         # Imag basis (Freq, N_FFT)
    Out_ptr,            # Output Magnitude: (total_frames, Freq)
    # Cache pointers
    stft_state_ptr,         # Cache: (num_cache_lines, CACHE_LEN) - stores last N_FFT-HOP samples
    cache_indices_ptr,      # (batch,) int32 - maps sequence to cache line index
    # Sequence mapping (from FastConformerConvMetadata)
    batch_ptr,              # (num_programs,) maps program_id -> sequence index
    time_chunk_offset_ptr,  # (num_programs,) maps program_id -> chunk index
    # Sequence boundaries
    query_start_loc_ptr,    # (batch+1,) cumulative positions in base units
    # Strides
    stride_rf, stride_rn,
    stride_cache_seq, stride_cache_sample,
    # Constants
    NUM_FREQS: tl.constexpr,
    N_FFT: tl.constexpr,
    HOP: tl.constexpr,
    CACHE_LEN: tl.constexpr,  # = N_FFT - HOP
    CACHE_BLOCK: tl.constexpr,  # Power of 2 >= CACHE_LEN for tl.arange
    MAG_POWER: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_FRAMES: tl.constexpr,  # Number of frames per chunk (like BLOCK_TIME in conv2d)
    TIME_FACTOR: tl.constexpr,  # Multiply query_start_loc to get sample positions
    OUTPUT_DIVISOR: tl.constexpr,  # Divide sample positions to get frame positions
    pad_slot_id: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
):
    """
    Varlen STFT magnitude kernel with cache support for streaming execution.
    
    Uses FastConformerConvMetadata-compatible inputs:
    - query_start_loc in base units, scaled by TIME_FACTOR to get sample positions
    - Frame positions computed as sample_positions // OUTPUT_DIVISOR
    
    Grid: (num_programs, freq_blocks)
    - program_id(0): sequence/chunk program (from batch_ptr/time_chunk_offset_ptr)
    - program_id(1): frequency block index
    
    Each program processes BLOCK_FRAMES consecutive frames of one sequence.
    This is analogous to BLOCK_TIME in the conv2d kernel.
    
    Virtual buffer model:
    - Virtual buffer = [cache(CACHE_LEN) | input(seq_len)]
    - Frame i reads virtual[i*HOP : i*HOP + N_FFT]
    - Cache contains last CACHE_LEN samples from previous call (or zeros for first call)
    
    After processing, the LAST chunk for each sequence updates the cache
    with the last CACHE_LEN samples from the virtual buffer.
    """
    # ==========================================================================
    # Step 1: Map program_id to (sequence_idx, chunk_offset)
    # ==========================================================================
    prog_id = tl.program_id(0)
    idx_seq = tl.load(batch_ptr + prog_id).to(tl.int64)
    chunk_offset = tl.load(time_chunk_offset_ptr + prog_id)
    
    pid_freq = tl.program_id(1)  # Frequency block index
    
    # Skip padding slots
    if USE_PAD_SLOT:
        if idx_seq == pad_slot_id:
            return
    
    # ==========================================================================
    # Step 2: Get cache index for this sequence
    # ==========================================================================
    cache_idx = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)
    
    if USE_PAD_SLOT:
        if cache_idx == pad_slot_id:
            return
    
    # ==========================================================================
    # Step 3: Get sequence boundaries using TIME_FACTOR and OUTPUT_DIVISOR
    # ==========================================================================
    # Load base positions and scale by TIME_FACTOR to get sample positions
    base_start = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    base_end = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    
    seq_sample_start = base_start * TIME_FACTOR
    seq_sample_end = base_end * TIME_FACTOR
    seq_len = seq_sample_end - seq_sample_start
    
    # Compute frame positions from sample positions using OUTPUT_DIVISOR
    seq_frame_start = seq_sample_start // OUTPUT_DIVISOR
    seq_frame_end = seq_sample_end // OUTPUT_DIVISOR
    seq_num_frames = seq_frame_end - seq_frame_start
    
    # ==========================================================================
    # Step 4: Compute chunk boundaries (like conv2d)
    # ==========================================================================
    frame_chunk_start = chunk_offset * BLOCK_FRAMES
    frame_chunk_end = tl.minimum(frame_chunk_start + BLOCK_FRAMES, seq_num_frames)
    chunk_len = frame_chunk_end - frame_chunk_start
    
    if chunk_len <= 0:
        return
    
    # ==========================================================================
    # Step 5: Load Basis Functions (Real and Imaginary) for this freq block
    # ==========================================================================
    freq_offsets = pid_freq * BLOCK_F + tl.arange(0, BLOCK_F)
    freq_mask = freq_offsets < NUM_FREQS
    
    sample_offsets = tl.arange(0, N_FFT)
    
    # w_real, w_imag: (BLOCK_F, N_FFT)
    w_real_ptrs = W_real_ptr + freq_offsets[:, None] * stride_rf + sample_offsets[None, :] * stride_rn
    w_imag_ptrs = W_imag_ptr + freq_offsets[:, None] * stride_rf + sample_offsets[None, :] * stride_rn
    
    w_real = tl.load(w_real_ptrs, mask=freq_mask[:, None], other=0.0)
    w_imag = tl.load(w_imag_ptrs, mask=freq_mask[:, None], other=0.0)
    
    # ==========================================================================
    # Step 6: Cache base pointer
    # ==========================================================================
    cache_base = stft_state_ptr + cache_idx * stride_cache_seq
    
    # ==========================================================================
    # Step 7: Process frames in this chunk
    # ==========================================================================
    # Virtual buffer = [cache(CACHE_LEN) | input(seq_len)]
    # Total virtual length = CACHE_LEN + seq_len
    # Frame i reads virtual[i*HOP : i*HOP + N_FFT]
    
    virtual_len = CACHE_LEN + seq_len
    
    for frame_local in tl.static_range(BLOCK_FRAMES):
        frame_seq = frame_chunk_start + frame_local  # Frame index within sequence
        valid_frame = frame_seq < seq_num_frames
        
        # Virtual window for this frame: [frame_seq * HOP, frame_seq * HOP + N_FFT)
        window_start = frame_seq * HOP
        virtual_pos = window_start + sample_offsets  # (N_FFT,) positions in virtual space
        
        # Bounds check for virtual buffer
        valid_pos = virtual_pos < virtual_len
        
        # Determine which samples come from cache vs input
        is_from_cache = virtual_pos < CACHE_LEN
        
        # === Cache reads ===
        # For samples from cache, use virtual_pos directly as cache index
        cache_read_pos = tl.where(is_from_cache, virtual_pos, 0)
        cache_vals = tl.load(
            cache_base + cache_read_pos * stride_cache_sample,
            mask=is_from_cache & valid_pos & valid_frame,
            other=0.0
        )
        
        # === Input reads ===
        # Input position = virtual_pos - CACHE_LEN
        input_read_pos = virtual_pos - CACHE_LEN
        # Clamp for safe access
        input_read_pos_safe = tl.where(
            is_from_cache | (input_read_pos < 0) | (input_read_pos >= seq_len),
            0,
            input_read_pos
        )
        input_valid = (~is_from_cache) & (input_read_pos >= 0) & (input_read_pos < seq_len)
        input_vals = tl.load(
            X_ptr + seq_sample_start + input_read_pos_safe,
            mask=input_valid & valid_frame,
            other=0.0
        )
        
        # Combine: one is zero (masked), the other has the value
        x_frame = cache_vals + input_vals
        
        # Compute Complex STFT (Dot Product)
        # x_frame: (N_FFT,), weights: (BLOCK_F, N_FFT)
        res_real = tl.sum(x_frame[None, :] * w_real, axis=1)
        res_imag = tl.sum(x_frame[None, :] * w_imag, axis=1)
        
        # Fused Magnitude & Power calculation
        mag_sq = (res_real * res_real) + (res_imag * res_imag)
        
        if MAG_POWER == 2.0:
            mag = mag_sq
        elif MAG_POWER == 1.0:
            mag = tl.sqrt(mag_sq)
        else:
            mag = tl.exp(tl.log(mag_sq + 1e-10) * (MAG_POWER / 2.0))
        
        # Store result - output layout is (total_frames, freqs)
        frame_global = seq_frame_start + frame_seq
        out_ptrs = Out_ptr + frame_global * NUM_FREQS + freq_offsets
        tl.store(out_ptrs, mag, mask=freq_mask & valid_frame)
    
    # ==========================================================================
    # Step 8: Update cache (only for last chunk, only freq block 0)
    # ==========================================================================
    # New cache = last CACHE_LEN samples of virtual buffer
    # = virtual[seq_len : seq_len + CACHE_LEN]
    
    is_last_chunk = (frame_chunk_end >= seq_num_frames)
    
    if is_last_chunk & (pid_freq == 0):
        cache_write_offsets = tl.arange(0, CACHE_BLOCK)
        cache_mask = cache_write_offsets < CACHE_LEN
        
        # Virtual position for each new cache sample
        # New cache = virtual[seq_len : seq_len + CACHE_LEN]
        virtual_pos_for_cache = seq_len + cache_write_offsets
        
        # Determine source: old cache or input
        is_from_old_cache = virtual_pos_for_cache < CACHE_LEN
        
        # === Load from old cache ===
        old_cache_pos = tl.where(is_from_old_cache, virtual_pos_for_cache, 0)
        old_cache_vals = tl.load(
            cache_base + old_cache_pos * stride_cache_sample,
            mask=cache_mask & is_from_old_cache,
            other=0.0
        )
        
        # === Load from input ===
        input_pos = virtual_pos_for_cache - CACHE_LEN
        input_pos_safe = tl.where(
            is_from_old_cache | (input_pos < 0) | (input_pos >= seq_len),
            0,
            input_pos
        )
        input_valid_cache = (~is_from_old_cache) & (input_pos >= 0) & (input_pos < seq_len)
        input_vals_cache = tl.load(
            X_ptr + seq_sample_start + input_pos_safe,
            mask=cache_mask & input_valid_cache,
            other=0.0
        )
        
        # Combine
        new_cache_vals = old_cache_vals + input_vals_cache
        
        # Store to cache
        tl.store(
            cache_base + cache_write_offsets * stride_cache_sample,
            new_cache_vals,
            mask=cache_mask
        )


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def stft_cached(
    x: torch.Tensor,                    # (total_samples,) - packed sequences
    w_real: torch.Tensor,               # (n_freqs, n_fft) - real basis
    w_imag: torch.Tensor,               # (n_freqs, n_fft) - imaginary basis
    out: torch.Tensor,                  # (total_frames, n_freqs) - pre-allocated output
    stft_state: torch.Tensor,           # (num_cache_lines, cache_len) - cache
    query_start_loc: torch.Tensor,      # (batch+1,) cumulative positions in base units
    cache_indices: torch.Tensor,        # (batch,) maps sequence to cache line
    n_fft: int,
    hop_length: int,
    mag_power: float = 2.0,
    pad_slot_id: int = PAD_SLOT_ID,
    block_f: int = 32,
    block_frames: int = 8,  # Default matches FastConformer's BLOCK_M
    time_factor: int = 1,   # Multiply query_start_loc to get sample positions
    output_divisor: int = 1,  # Divide sample positions to get frame positions
    metadata=None,                      # Optional: FastConformerConvMetadata-compatible
) -> torch.Tensor:
    """
    Varlen STFT magnitude with cache support for streaming.
    
    Compatible with FastConformerConvMetadata:
    - Uses batch_ptr and token_chunk_offset_ptr from metadata
    - Applies time_factor and output_divisor to compute positions
    
    Position computation:
    - sample_position = query_start_loc * time_factor
    - frame_position = sample_position // output_divisor
    
    For typical STFT usage where query_start_loc is in samples:
    - time_factor = 1
    - output_divisor = hop_length
    
    For FastConformer where query_start_loc is in tokens:
    - time_factor = samples_per_token (e.g., hop_length if tokens = frames)
    - output_divisor = hop_length
    
    Args:
        x: Packed input samples (total_samples,)
        w_real: Real DFT basis with window applied (n_freqs, n_fft)
        w_imag: Imaginary DFT basis with window applied (n_freqs, n_fft)
        out: Pre-allocated output tensor (total_frames, n_freqs)
        stft_state: Cache tensor (num_cache_lines, cache_len) where cache_len = n_fft - hop_length
                   Stores the last samples from previous chunk. Updated in-place after processing.
                   Initialize to zeros before first call.
        query_start_loc: Cumulative positions in base units (batch+1,)
        cache_indices: Maps each sequence to cache line (batch,)
        n_fft: FFT size
        hop_length: Hop length between frames
        mag_power: Power for magnitude (1.0 = magnitude, 2.0 = power spectrum)
        pad_slot_id: Padding slot ID for skipping invalid sequences
        block_f: Frequency block size
        block_frames: Number of frames per chunk (default 8, matches FastConformer BLOCK_M)
        time_factor: Multiplier for query_start_loc to get sample positions
        output_divisor: Divisor for sample positions to get frame positions
        metadata: Optional FastConformerConvMetadata with pre-computed batch_ptr, 
                  token_chunk_offset_ptr, and optionally nums_dict for CUDA graph compatibility
    
    Returns:
        Output tensor (total_frames, n_freqs)
    """
    n_freqs = w_real.shape[0]
    cache_len = n_fft - hop_length
    
    # Get batch_ptr and time_chunk_offset_ptr from metadata or compute
    if metadata is not None:
        # Use pre-computed values from FastConformerConvMetadata (no CPU blocking)
        batch_ptr = metadata.batch_ptr
        time_chunk_offset_ptr = metadata.token_chunk_offset_ptr
        
        # Get num_programs from metadata if available
        if hasattr(metadata, 'nums_dict') and metadata.nums_dict is not None:
            # FastConformerConvMetadata stores this in nums_dict[BLOCK_M]
            if block_frames in metadata.nums_dict:
                num_programs = metadata.nums_dict[block_frames]['tot']
            else:
                # Fall back to the first available block size
                first_key = next(iter(metadata.nums_dict.keys()))
                num_programs = metadata.nums_dict[first_key]['mlist_len']
        elif hasattr(metadata, 'num_programs'):
            num_programs = metadata.num_programs
        else:
            # Count non-padding entries in batch_ptr
            num_programs = (batch_ptr != pad_slot_id).sum().item()
    else:
        # Compute on the fly (CPU blocking - not safe for CUDA graphs)
        query_start_loc_cpu = query_start_loc.cpu()
        
        # Scale by time_factor to get sample lengths
        seqlens_samples = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]) * time_factor
        
        batch_size = len(seqlens_samples)
        
        # Number of frames per sequence:
        # num_frames = (seq_len_samples) // output_divisor
        seqlens_frames = seqlens_samples // output_divisor
        
        total_frames = seqlens_frames.sum().item()
        
        if total_frames == 0:
            return out
        
        # Build program mapping: each program handles block_frames frames
        # Use ceiling division to ensure all frames are covered
        batch_list = []
        chunk_offset_list = []
        
        for seq_idx, seq_frames in enumerate(seqlens_frames.numpy()):
            seq_frames = int(seq_frames)
            num_chunks = (seq_frames + block_frames - 1) // block_frames if seq_frames > 0 else 1
            if num_chunks > 0:
                batch_list.extend([seq_idx] * num_chunks)
                chunk_offset_list.extend(range(num_chunks))
        
        num_programs = len(batch_list)
        
        if num_programs == 0:
            return out
        
        batch_ptr = torch.tensor(batch_list, dtype=torch.int32, device=x.device)
        time_chunk_offset_ptr = torch.tensor(chunk_offset_list, dtype=torch.int32, device=x.device)
    
    # Get strides
    stride_cache_seq, stride_cache_sample = stft_state.stride()
    
    # Launch kernel
    freq_blocks = triton.cdiv(n_freqs, block_f)
    grid = (num_programs, freq_blocks)
    
    # Compute CACHE_BLOCK as next power of 2 >= cache_len (required by tl.arange)
    cache_block = _next_power_of_2(cache_len)
    
    stft_cached_kernel[grid](
        x, w_real, w_imag, out,
        stft_state, cache_indices,
        batch_ptr, time_chunk_offset_ptr,
        query_start_loc,
        w_real.stride(0), w_real.stride(1),
        stride_cache_seq, stride_cache_sample,
        NUM_FREQS=n_freqs,
        N_FFT=n_fft,
        HOP=hop_length,
        CACHE_LEN=cache_len,
        CACHE_BLOCK=cache_block,
        MAG_POWER=mag_power,
        BLOCK_F=block_f,
        BLOCK_FRAMES=block_frames,
        TIME_FACTOR=time_factor,
        OUTPUT_DIVISOR=output_divisor,
        pad_slot_id=pad_slot_id,
        USE_PAD_SLOT=pad_slot_id is not None,
    )
    
    return out
