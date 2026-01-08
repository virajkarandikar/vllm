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
    # Sequence mapping (computed on CPU, passed to kernel)
    batch_ptr,              # (num_programs,) maps program_id -> sequence index
    time_chunk_offset_ptr,  # (num_programs,) maps program_id -> chunk index (not frame offset)
    # Sequence boundaries
    query_start_loc_ptr,     # (batch+1,) cumulative sample positions
    query_start_loc_out_ptr, # (batch+1,) cumulative frame positions
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
    BLOCK_FRAMES: tl.constexpr,
    pad_slot_id: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
):
    """
    Varlen STFT magnitude kernel with cache support for streaming execution.
    
    Grid: (num_programs, freq_blocks)
    - program_id(0): sequence/frame chunk program (from batch_ptr/frame_chunk_offset_ptr)
    - program_id(1): frequency block index
    
    Cache layout: (num_cache_lines, CACHE_LEN) where CACHE_LEN = N_FFT - HOP
    - Stores the last CACHE_LEN samples from previous chunk
    - Cache is ALWAYS used: virtual input = [cache | new_samples]
    - For first call, initialize cache to zeros (gives zero-padding effect)
    - After processing, updates cache with last CACHE_LEN samples
    
    Virtual addressing (always with cache):
    - Virtual position 0 to CACHE_LEN-1: read from cache
    - Virtual position CACHE_LEN onwards: read from input (offset by CACHE_LEN)
    """
    # ==========================================================================
    # Step 1: Map program_id to (sequence_idx, frame_chunk_start)
    # ==========================================================================
    prog_id = tl.program_id(0)
    idx_seq = tl.load(batch_ptr + prog_id).to(tl.int64)
    chunk_offset = tl.load(time_chunk_offset_ptr + prog_id)
    frame_chunk_start = chunk_offset * BLOCK_FRAMES  # Convert chunk index to frame offset
    
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
    # Step 3: Get sequence boundaries
    # ==========================================================================
    seq_sample_start = tl.load(query_start_loc_ptr + idx_seq)
    seq_sample_end = tl.load(query_start_loc_ptr + idx_seq + 1)
    seq_len = seq_sample_end - seq_sample_start
    
    seq_frame_start = tl.load(query_start_loc_out_ptr + idx_seq)
    seq_frame_end = tl.load(query_start_loc_out_ptr + idx_seq + 1)
    seq_num_frames = seq_frame_end - seq_frame_start
    
    # ==========================================================================
    # Step 4: Load Basis Functions (Real and Imaginary) for this freq block
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
    # Step 5: Cache base pointer
    # ==========================================================================
    cache_base = stft_state_ptr + cache_idx * stride_cache_seq
    
    # ==========================================================================
    # Step 6: Process frames in this chunk
    # ==========================================================================
    # Virtual space is ALWAYS [cache(CACHE_LEN) | input(seq_len)]
    # - Positions [0, CACHE_LEN): from cache
    # - Positions [CACHE_LEN, CACHE_LEN + seq_len): from input
    
    for frame_local in tl.static_range(BLOCK_FRAMES):
        frame_seq = frame_chunk_start + frame_local  # Frame index within sequence
        valid_frame = frame_seq < seq_num_frames
        
        # Virtual window for this frame: [frame_seq * HOP, frame_seq * HOP + N_FFT)
        window_start = frame_seq * HOP
        virtual_pos = window_start + sample_offsets  # (N_FFT,) - positions in virtual space
        
        # Determine which samples come from cache vs input
        is_from_cache = virtual_pos < CACHE_LEN
        
        # === Cache reads ===
        # For samples from cache, use virtual_pos directly as cache index
        # Clamp positions for safe memory access (masked out for input samples)
        cache_read_pos = tl.where(is_from_cache, virtual_pos, 0)
        cache_vals = tl.load(
            cache_base + cache_read_pos * stride_cache_sample,
            mask=is_from_cache & valid_frame,
            other=0.0
        )
        
        # === Input reads ===
        # Input position = virtual_pos - CACHE_LEN
        input_read_pos = virtual_pos - CACHE_LEN
        # Clamp for safe access (masked out for cache samples)
        input_read_pos_safe = tl.where(is_from_cache, 0, input_read_pos)
        input_vals = tl.load(
            X_ptr + seq_sample_start + input_read_pos_safe,
            mask=(~is_from_cache) & valid_frame,
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
    # Step 7: Update cache with last CACHE_LEN samples from input
    # Only the last chunk (for this freq block, but we only need pid_freq==0) 
    # should update the cache. We check if this chunk contains the last frame.
    # ==========================================================================
    last_frame_in_chunk = frame_chunk_start + BLOCK_FRAMES - 1
    is_last_chunk = (last_frame_in_chunk >= seq_num_frames - 1) | (frame_chunk_start + BLOCK_FRAMES > seq_num_frames)
    
    # Only one frequency block should update the cache to avoid races
    if is_last_chunk & (pid_freq == 0):
        # Store last CACHE_LEN samples from input to cache
        # These are input samples at positions [seq_len - CACHE_LEN, seq_len)
        # Use CACHE_BLOCK (power of 2) for tl.arange and mask with CACHE_LEN
        cache_write_offsets = tl.arange(0, CACHE_BLOCK)
        cache_mask = cache_write_offsets < CACHE_LEN
        input_tail_start = seq_sample_start + seq_len - CACHE_LEN
        
        # Load last CACHE_LEN samples from input
        tail_samples = tl.load(
            X_ptr + input_tail_start + cache_write_offsets,
            mask=cache_mask,
            other=0.0
        )
        
        # Store to cache
        tl.store(
            cache_base + cache_write_offsets * stride_cache_sample,
            tail_samples,
            mask=cache_mask
        )


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    # Bit manipulation: subtract 1, then find next power of 2
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
    query_start_loc: torch.Tensor,      # (batch+1,) cumulative sample positions
    cache_indices: torch.Tensor,        # (batch,) maps sequence to cache line
    n_fft: int,
    hop_length: int,
    mag_power: float = 2.0,
    pad_slot_id: int = PAD_SLOT_ID,
    block_f: int = 32,
    block_frames: int = 16,
    metadata=None,                      # Optional: provides pre-computed batch_ptr, etc.
) -> torch.Tensor:
    """
    Varlen STFT magnitude with cache support for streaming.
    
    Simplified version that ALWAYS uses cache:
    - stft_state stores the last (n_fft - hop_length) samples from previous chunk
    - cache_indices maps each sequence to its cache line
    - Initialize cache to zeros before first call (gives zero-padding effect)
    
    For proper streaming, input lengths should be divisible by hop_length.
    This ensures: num_frames = seq_len // hop_length
    
    If metadata is provided, uses pre-computed batch_ptr/time_chunk_offset_ptr from it
    (no CPU blocking). Otherwise computes on the fly (CPU blocking - not for CUDA graphs).
    
    Args:
        x: Packed input samples (total_samples,)
        w_real: Real DFT basis with window applied (n_freqs, n_fft)
        w_imag: Imaginary DFT basis with window applied (n_freqs, n_fft)
        out: Pre-allocated output tensor (total_frames, n_freqs)
        stft_state: Cache tensor (num_cache_lines, cache_len) where cache_len = n_fft - hop_length
                   Stores the last samples from previous chunk. Updated in-place after processing.
                   Initialize to zeros before first call.
        query_start_loc: Cumulative sample positions (batch+1,)
        cache_indices: Maps each sequence to cache line (batch,)
        n_fft: FFT size
        hop_length: Hop length between frames
        mag_power: Power for magnitude (1.0 = magnitude, 2.0 = power spectrum)
        pad_slot_id: Padding slot ID for skipping invalid sequences
        block_f: Frequency block size
        block_frames: Number of frames per program
        metadata: Optional metadata with pre-computed batch_ptr, time_chunk_offset_ptr,
                  query_start_loc_out, num_programs (for CUDA graph compatibility)
    
    Returns:
        Output tensor (total_frames, n_freqs)
    """
    n_freqs = w_real.shape[0]
    cache_len = n_fft - hop_length
    
    assert stft_state.shape[1] == cache_len, (
        f"stft_state cache_len mismatch: {stft_state.shape[1]} vs expected {cache_len}"
    )
    
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
        seqlens_samples = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        
        batch_size = len(seqlens_samples)
        
        # Number of frames per sequence (cache always used):
        # num_frames = seq_len // hop_length
        seqlens_frames = seqlens_samples // hop_length
        
        # Build output query_start_loc
        query_start_loc_out_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        query_start_loc_out_cpu[1:] = torch.cumsum(seqlens_frames, dim=0)
        total_frames = query_start_loc_out_cpu[-1].item()
        
        if total_frames == 0:
            return out
        
        # Build program mapping: each program handles BLOCK_FRAMES frames of one sequence
        batch_list = []
        chunk_offset_list = []
        
        for seq_idx, seq_frames in enumerate(seqlens_frames.numpy()):
            num_chunks = int(np.ceil(seq_frames / block_frames)) if seq_frames > 0 else 0
            if num_chunks > 0:
                batch_list.extend([seq_idx] * num_chunks)
                chunk_offset_list.extend(range(num_chunks))  # Chunk indices: 0, 1, 2, ...
        
        num_programs = len(batch_list)
        
        if num_programs == 0:
            return out
        
        batch_ptr = torch.tensor(batch_list, dtype=torch.int32, device=x.device)
        time_chunk_offset_ptr = torch.tensor(chunk_offset_list, dtype=torch.int32, device=x.device)
        query_start_loc_out = query_start_loc_out_cpu.to(x.device)
    
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
        query_start_loc, query_start_loc_out,
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
        pad_slot_id=pad_slot_id,
        USE_PAD_SLOT=pad_slot_id is not None,
    )
    
    return out
