# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 NVIDIA

"""
Causal 2D Convolution with kernel_size=(3,3), stride=(2,2) for mel spectrogram processing.

This kernel is designed for audio preprocessing (e.g., FastConformer subsampling) where:
- Time dimension: requires caching for causality, kernel=3, stride=2
- Frequency dimension: fixed input size (80 or padded), no caching needed
- Maximum frequency input size: 82 (freq_out <= 40)

Kernel: (3, 3) over (time, frequency)
Stride: (2, 2) in both dimensions  
Cache: 1 time frame (kernel_t - stride_t = 3 - 2 = 1)

Supports two modes:
1. Depthwise convolution (in_channels == out_channels):
   - Input: (C, T, F_in), Weight: (C, 3, 3), Output: (C, T_out, F_out)
   - Each output channel only depends on the corresponding input channel
   - Equivalent to nn.Conv2d with groups=C
   
2. Broadcast convolution (in_channels == 1):
   - Input: (1, T, F_in), Weight: (C_out, 3, 3), Output: (C_out, T_out, F_out)
   - All output channels read from the single input channel
   - Equivalent to nn.Conv2d with in_channels=1, out_channels=C_out

Output dimensions:
    - T_out = (T + 1) // 2  (with cache providing the -1 time position)
    - F_out = (F_in - 3) // 2 + 1

For each output position (c_out, t_out, f_out):
    out[c_out, t_out, f_out] = sum over (kt, kf) of:
        w[c_out, kt, kf] * x[c_in, t_out*2 + kt - 1, f_out*2 + kf]
    Where c_in = c_out for depthwise, c_in = 0 for broadcast.
    Time position -1 comes from cache.

Optimization notes:
    - BLOCK_F=64 covers all frequencies at once (freq_out <= 40 for freq_in <= 82)
    - No frequency loop needed - all frequencies processed in parallel
    - BLOCK_C=64 for efficient channel parallelism
"""

from typing import Optional

import numpy as np
import torch

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.triton_utils import tl, triton

# Maximum frequency dimensions - kernel assumes freq_in <= 82
MAX_FREQ_IN = 82
MAX_FREQ_OUT = (MAX_FREQ_IN - 3) // 2 + 1  # = 40


@triton.jit()
def _causal_conv2d_k3s2_fwd_kernel(
    # Pointers to matrices
    x_ptr,  # (C_in, cu_time, F_in) input
    w_ptr,  # (C_out, 3, 3) weights for each output channel
    bias_ptr,  # (C_out,) optional bias
    cache_ptr,  # (num_cache_lines, C_in, 1, F_in) conv cache - 1 time frame
    cache_indices_ptr,  # (batch,) maps sequence to cache line
    has_initial_state_ptr,  # (batch,) whether sequence has cached state
    query_start_loc_ptr,  # (batch + 1,) cumsum of input time lengths
    output_start_loc_ptr,  # (batch + 1,) cumsum of output time lengths
    batch_ptr,  # (num_programs,) maps program_id to sequence index
    time_chunk_offset_ptr,  # (num_programs,) maps program_id to output time chunk
    o_ptr,  # (C_out, cu_time_out, F_out) output
    # Matrix dimensions
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    freq_in: tl.constexpr,
    freq_out: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides for x: (C_in, T, F)
    stride_x_c: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_f: tl.constexpr,
    # Strides for w: (C_out, 3, 3)
    stride_w_c: tl.constexpr,
    stride_w_kt: tl.constexpr,
    stride_w_kf: tl.constexpr,
    # Strides for cache: (num_cache_lines, C_in, 1, F_in)
    stride_cache_seq: tl.constexpr,
    stride_cache_c: tl.constexpr,
    stride_cache_t: tl.constexpr,
    stride_cache_f: tl.constexpr,
    # Strides for output: (C_out, T_out, F_out)
    stride_o_c: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_f: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    IS_DEPTHWISE: tl.constexpr,  # True if in_channels == out_channels (1:1 mapping)
    BLOCK_T: tl.constexpr,  # output time positions per chunk
    BLOCK_C: tl.constexpr,  # channels per block
    BLOCK_F: tl.constexpr,  # frequency outputs per block (covers all freq_out)
):
    """
    Forward kernel for causal 2D conv with kernel=(3,3), stride=(2,2).

    Grid: (num_programs, ceil(out_channels / BLOCK_C))
    - program_id(0): indexes into batch_ptr/time_chunk_offset_ptr
    - program_id(1): output channel block index

    Each program processes BLOCK_T consecutive output time positions for
    BLOCK_C output channels, computing all F_out frequency outputs in parallel.
    
    Optimizations:
    - All frequency outputs processed in parallel (no freq loop)
    - Uses 2D (BLOCK_C, BLOCK_F) tensor operations
    - Assumes freq_out <= BLOCK_F (no need to tile over frequency)
    
    Supports two modes:
    - IS_DEPTHWISE=True: in_channels == out_channels, 1:1 channel mapping
    - IS_DEPTHWISE=False: in_channels == 1, all outputs read from channel 0
    """
    # Get sequence index and chunk offset for this program
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(time_chunk_offset_ptr + tl.program_id(0))

    if idx_seq == pad_slot_id:
        return

    # Output channel indices for this block: (BLOCK_C,)
    idx_out_channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = idx_out_channels < out_channels
    
    # Output frequency indices: (BLOCK_F,) - process all frequencies at once
    idx_f_out = tl.arange(0, BLOCK_F)
    mask_f = idx_f_out < freq_out
    
    # 2D mask for (BLOCK_C, BLOCK_F)
    mask_cf = mask_c[:, None] & mask_f[None, :]
    
    # Input channel indices: same as output for depthwise, always 0 for broadcast
    if IS_DEPTHWISE:
        idx_in_channels = idx_out_channels
    else:
        # in_channels == 1, all outputs read from input channel 0
        idx_in_channels = tl.zeros((BLOCK_C,), dtype=tl.int32)

    # Get sequence boundaries (input time)
    in_t_start = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    in_t_end = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    in_t_len = in_t_end - in_t_start

    # Output time length: (T_in + 1) // 2 (cache provides position -1)
    out_t_len = (in_t_len + 1) // 2

    # Get output sequence start position
    out_t_start = tl.load(output_start_loc_ptr + idx_seq).to(tl.int64)

    # Get cache index for this sequence
    cache_idx = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)

    if USE_PAD_SLOT:
        if cache_idx == pad_slot_id:
            return

    # Preload weights: w[c_out, kt, kf] for all 9 kernel positions
    # Shape: (BLOCK_C,) for each of 9 positions, will broadcast to (BLOCK_C, BLOCK_F)
    w_base = w_ptr + idx_out_channels * stride_w_c

    w00 = tl.load(w_base + 0 * stride_w_kt + 0 * stride_w_kf, mask=mask_c, other=0.0)
    w01 = tl.load(w_base + 0 * stride_w_kt + 1 * stride_w_kf, mask=mask_c, other=0.0)
    w02 = tl.load(w_base + 0 * stride_w_kt + 2 * stride_w_kf, mask=mask_c, other=0.0)
    w10 = tl.load(w_base + 1 * stride_w_kt + 0 * stride_w_kf, mask=mask_c, other=0.0)
    w11 = tl.load(w_base + 1 * stride_w_kt + 1 * stride_w_kf, mask=mask_c, other=0.0)
    w12 = tl.load(w_base + 1 * stride_w_kt + 2 * stride_w_kf, mask=mask_c, other=0.0)
    w20 = tl.load(w_base + 2 * stride_w_kt + 0 * stride_w_kf, mask=mask_c, other=0.0)
    w21 = tl.load(w_base + 2 * stride_w_kt + 1 * stride_w_kf, mask=mask_c, other=0.0)
    w22 = tl.load(w_base + 2 * stride_w_kt + 2 * stride_w_kf, mask=mask_c, other=0.0)

    # Preload bias if present (indexed by output channel)
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + idx_out_channels, mask=mask_c, other=0.0).to(
            tl.float32
        )
    else:
        bias_val = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Input frequency positions for all output frequencies: (BLOCK_F,)
    # f_in = f_out * 2 + kf, for kf in {0, 1, 2}
    f_in0 = idx_f_out * 2 + 0  # (BLOCK_F,)
    f_in1 = idx_f_out * 2 + 1  # (BLOCK_F,)
    f_in2 = idx_f_out * 2 + 2  # (BLOCK_F,)

    # Base pointers for input/cache (indexed by input channel): (BLOCK_C,)
    cache_base = (
        cache_ptr
        + cache_idx * stride_cache_seq
        + idx_in_channels * stride_cache_c
    )

    x_base = (
        x_ptr
        + in_t_start * stride_x_t
        + idx_in_channels * stride_x_c
    )

    # Base pointer for output (indexed by output channel): (BLOCK_C,)
    o_base = (
        o_ptr
        + out_t_start * stride_o_t
        + idx_out_channels * stride_o_c
    )

    # First chunk (chunk_offset == 0): determine if we have cached state
    if chunk_offset == 0:
        # Load cached frame (time position -1) if available
        has_state = tl.load(has_initial_state_ptr + idx_seq).to(tl.int1)
    else:
        has_state = True  # For non-first chunks, we always have prior data from x

    # Compute output time position range for this chunk
    t_out_start = chunk_offset * BLOCK_T

    # Process output time positions in this chunk
    for t_idx in range(BLOCK_T):
        t_out = t_out_start + t_idx
        # Use conditional guard instead of break (Triton doesn't support break)
        if t_out < out_t_len:
            # Accumulator: (BLOCK_C, BLOCK_F)
            acc = bias_val[:, None] + tl.zeros((BLOCK_C, BLOCK_F), dtype=tl.float32)

            # Input time positions: t_out*2 + kt - 1 for kt in 0,1,2
            t_in0 = t_out * 2 - 1
            t_in1 = t_out * 2
            t_in2 = t_out * 2 + 1

            # Create 2D pointers: (BLOCK_C, BLOCK_F)
            # For cache: cache_base[:, None] + f_in[None, :] * stride_cache_f
            # For x: x_base[:, None] + t * stride_x_t + f_in[None, :] * stride_x_f

            # Load x values for kt=0 (t_in0 may be -1)
            if t_in0 < 0:
                # Load from cache: (BLOCK_C, BLOCK_F)
                if has_state:
                    cache_ptr_2d_0 = cache_base[:, None] + f_in0[None, :] * stride_cache_f
                    cache_ptr_2d_1 = cache_base[:, None] + f_in1[None, :] * stride_cache_f
                    cache_ptr_2d_2 = cache_base[:, None] + f_in2[None, :] * stride_cache_f
                    x00 = tl.load(cache_ptr_2d_0, mask=mask_cf, other=0.0)
                    x01 = tl.load(cache_ptr_2d_1, mask=mask_cf, other=0.0)
                    x02 = tl.load(cache_ptr_2d_2, mask=mask_cf, other=0.0)
                else:
                    x00 = tl.zeros((BLOCK_C, BLOCK_F), dtype=x_ptr.dtype.element_ty)
                    x01 = tl.zeros((BLOCK_C, BLOCK_F), dtype=x_ptr.dtype.element_ty)
                    x02 = tl.zeros((BLOCK_C, BLOCK_F), dtype=x_ptr.dtype.element_ty)
            else:
                x_ptr_2d_t0_f0 = x_base[:, None] + t_in0 * stride_x_t + f_in0[None, :] * stride_x_f
                x_ptr_2d_t0_f1 = x_base[:, None] + t_in0 * stride_x_t + f_in1[None, :] * stride_x_f
                x_ptr_2d_t0_f2 = x_base[:, None] + t_in0 * stride_x_t + f_in2[None, :] * stride_x_f
                x00 = tl.load(x_ptr_2d_t0_f0, mask=mask_cf, other=0.0)
                x01 = tl.load(x_ptr_2d_t0_f1, mask=mask_cf, other=0.0)
                x02 = tl.load(x_ptr_2d_t0_f2, mask=mask_cf, other=0.0)

            # Load x values for kt=1 (t_in1): (BLOCK_C, BLOCK_F)
            x_ptr_2d_t1_f0 = x_base[:, None] + t_in1 * stride_x_t + f_in0[None, :] * stride_x_f
            x_ptr_2d_t1_f1 = x_base[:, None] + t_in1 * stride_x_t + f_in1[None, :] * stride_x_f
            x_ptr_2d_t1_f2 = x_base[:, None] + t_in1 * stride_x_t + f_in2[None, :] * stride_x_f
            x10 = tl.load(x_ptr_2d_t1_f0, mask=mask_cf, other=0.0)
            x11 = tl.load(x_ptr_2d_t1_f1, mask=mask_cf, other=0.0)
            x12 = tl.load(x_ptr_2d_t1_f2, mask=mask_cf, other=0.0)

            # Load x values for kt=2 (t_in2): (BLOCK_C, BLOCK_F)
            x_ptr_2d_t2_f0 = x_base[:, None] + t_in2 * stride_x_t + f_in0[None, :] * stride_x_f
            x_ptr_2d_t2_f1 = x_base[:, None] + t_in2 * stride_x_t + f_in1[None, :] * stride_x_f
            x_ptr_2d_t2_f2 = x_base[:, None] + t_in2 * stride_x_t + f_in2[None, :] * stride_x_f
            x20 = tl.load(x_ptr_2d_t2_f0, mask=mask_cf, other=0.0)
            x21 = tl.load(x_ptr_2d_t2_f1, mask=mask_cf, other=0.0)
            x22 = tl.load(x_ptr_2d_t2_f2, mask=mask_cf, other=0.0)

            # Compute convolution: broadcast weights (BLOCK_C,) -> (BLOCK_C, BLOCK_F)
            acc += w00[:, None] * x00 + w01[:, None] * x01 + w02[:, None] * x02
            acc += w10[:, None] * x10 + w11[:, None] * x11 + w12[:, None] * x12
            acc += w20[:, None] * x20 + w21[:, None] * x21 + w22[:, None] * x22

            # Apply activation if specified
            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))

            # Store output: (BLOCK_C, BLOCK_F)
            o_ptr_2d = o_base[:, None] + t_out * stride_o_t + idx_f_out[None, :] * stride_o_f
            tl.store(o_ptr_2d, acc, mask=mask_cf)

    # Update cache with last input time frame for next batch
    # Cache stores input[T-1, :] which becomes position -1 for next batch
    # IMPORTANT: This must happen AFTER reading cache for t_out=0 above
    # For broadcast mode (in_channels=1), only first output block updates cache
    should_update_cache = chunk_offset == 0
    if not IS_DEPTHWISE:
        # In broadcast mode, only program_id(1)==0 should update the single input channel cache
        should_update_cache = should_update_cache and (tl.program_id(1) == 0)
    
    if should_update_cache:
        tl.debug_barrier()
        last_t = in_t_len - 1
        # For broadcast mode, mask should cover the single input channel
        if IS_DEPTHWISE:
            cache_mask = mask_c
        else:
            # in_channels=1, only one value to store (no mask needed, always valid)
            cache_mask = tl.arange(0, BLOCK_C) < 1
        
        # Update cache for all input frequencies at once
        # Cache shape: (BLOCK_C,) -> store freq_in values
        idx_f_cache = tl.arange(0, BLOCK_F)
        mask_f_cache = idx_f_cache < freq_in
        mask_cache_2d = cache_mask[:, None] & mask_f_cache[None, :]
        
        cache_store_ptr = cache_base[:, None] + idx_f_cache[None, :] * stride_cache_f
        x_load_ptr = x_base[:, None] + last_t * stride_x_t + idx_f_cache[None, :] * stride_x_f
        val = tl.load(x_load_ptr, mask=mask_cache_2d, other=0.0)
        tl.store(cache_store_ptr, val, mask=mask_cache_2d)


@triton.jit()
def _causal_conv2d_k3s2_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, C_in, T, F_in) or (C_in, cu_time, F_in) for varlen
    w_ptr,  # (C_out, 3, 3) weights
    bias_ptr,  # (C_out,) optional bias
    cache_ptr,  # (num_cache_lines, C_in, 1, F_in) conv cache
    cache_indices_ptr,  # (batch,) maps sequence to cache line
    query_start_loc_ptr,  # (batch + 1,) for varlen mode
    o_ptr,  # output
    # Matrix dimensions
    batch: int,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    time_len: tl.constexpr,  # input time length per sequence
    freq_in: tl.constexpr,
    freq_out: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides for x
    stride_x_b: tl.constexpr,
    stride_x_c: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_f: tl.constexpr,
    # Strides for w
    stride_w_c: tl.constexpr,
    stride_w_kt: tl.constexpr,
    stride_w_kf: tl.constexpr,
    # Strides for cache
    stride_cache_seq: tl.constexpr,
    stride_cache_c: tl.constexpr,
    stride_cache_t: tl.constexpr,
    stride_cache_f: tl.constexpr,
    # Strides for output
    stride_o_b: tl.constexpr,
    stride_o_c: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_f: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    IS_DEPTHWISE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_F: tl.constexpr,  # frequency outputs per block (covers all freq_out)
):
    """
    Update kernel for decode mode (small number of time frames).

    Grid: (batch, ceil(out_channels / BLOCK_C))
    
    Optimizations:
    - All frequency outputs processed in parallel (no freq loop)
    - Uses 2D (BLOCK_C, BLOCK_F) tensor operations
    - Assumes freq_out <= BLOCK_F (no need to tile over frequency)
    
    Supports two modes:
    - IS_DEPTHWISE=True: in_channels == out_channels, 1:1 channel mapping
    - IS_DEPTHWISE=False: in_channels == 1, all outputs read from channel 0
    """
    idx_batch = tl.program_id(0)
    if idx_batch >= batch:
        return

    # Output channel indices for this block: (BLOCK_C,)
    idx_out_channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = idx_out_channels < out_channels
    
    # Output frequency indices: (BLOCK_F,) - process all frequencies at once
    idx_f_out = tl.arange(0, BLOCK_F)
    mask_f = idx_f_out < freq_out
    
    # 2D mask for (BLOCK_C, BLOCK_F)
    mask_cf = mask_c[:, None] & mask_f[None, :]
    
    # Input channel indices
    if IS_DEPTHWISE:
        idx_in_channels = idx_out_channels
    else:
        idx_in_channels = tl.zeros((BLOCK_C,), dtype=tl.int32)

    # Get cache index
    cache_idx = tl.load(cache_indices_ptr + idx_batch).to(tl.int64)

    if USE_PAD_SLOT:
        if cache_idx == pad_slot_id:
            return

    # Handle varlen vs fixed length
    if IS_VARLEN:
        t_start = tl.load(query_start_loc_ptr + idx_batch).to(tl.int64)
        t_end = tl.load(query_start_loc_ptr + idx_batch + 1).to(tl.int64)
        actual_t_len = t_end - t_start
        x_offset = t_start * stride_x_t
        out_t_len = (actual_t_len + 1) // 2
        out_t_start = (t_start + 1) // 2  # Approximate - should use output_start_loc
        o_offset = out_t_start * stride_o_t
    else:
        actual_t_len = time_len
        x_offset = idx_batch * stride_x_b
        out_t_len = (actual_t_len + 1) // 2
        o_offset = idx_batch * stride_o_b

    if actual_t_len == 0:
        return

    # Load weights (indexed by output channel): (BLOCK_C,)
    w_base = w_ptr + idx_out_channels * stride_w_c
    w00 = tl.load(w_base + 0 * stride_w_kt + 0 * stride_w_kf, mask=mask_c, other=0.0)
    w01 = tl.load(w_base + 0 * stride_w_kt + 1 * stride_w_kf, mask=mask_c, other=0.0)
    w02 = tl.load(w_base + 0 * stride_w_kt + 2 * stride_w_kf, mask=mask_c, other=0.0)
    w10 = tl.load(w_base + 1 * stride_w_kt + 0 * stride_w_kf, mask=mask_c, other=0.0)
    w11 = tl.load(w_base + 1 * stride_w_kt + 1 * stride_w_kf, mask=mask_c, other=0.0)
    w12 = tl.load(w_base + 1 * stride_w_kt + 2 * stride_w_kf, mask=mask_c, other=0.0)
    w20 = tl.load(w_base + 2 * stride_w_kt + 0 * stride_w_kf, mask=mask_c, other=0.0)
    w21 = tl.load(w_base + 2 * stride_w_kt + 1 * stride_w_kf, mask=mask_c, other=0.0)
    w22 = tl.load(w_base + 2 * stride_w_kt + 2 * stride_w_kf, mask=mask_c, other=0.0)

    # Load bias (indexed by output channel)
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + idx_out_channels, mask=mask_c, other=0.0).to(
            tl.float32
        )
    else:
        bias_val = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Input frequency positions for all output frequencies: (BLOCK_F,)
    f_in0 = idx_f_out * 2 + 0
    f_in1 = idx_f_out * 2 + 1
    f_in2 = idx_f_out * 2 + 2

    # Base pointers for input/cache (indexed by input channel): (BLOCK_C,)
    cache_base = cache_ptr + cache_idx * stride_cache_seq + idx_in_channels * stride_cache_c
    x_base = x_ptr + x_offset + idx_in_channels * stride_x_c
    # Base pointer for output (indexed by output channel): (BLOCK_C,)
    o_base = o_ptr + o_offset + idx_out_channels * stride_o_c

    # Process output time positions
    for t_out in range(out_t_len):
        # Accumulator: (BLOCK_C, BLOCK_F)
        acc = bias_val[:, None] + tl.zeros((BLOCK_C, BLOCK_F), dtype=tl.float32)

        t_in0 = t_out * 2 - 1
        t_in1 = t_out * 2
        t_in2 = t_out * 2 + 1

        # Load x for kt=0: (BLOCK_C, BLOCK_F)
        if t_in0 < 0:
            cache_ptr_2d_0 = cache_base[:, None] + f_in0[None, :] * stride_cache_f
            cache_ptr_2d_1 = cache_base[:, None] + f_in1[None, :] * stride_cache_f
            cache_ptr_2d_2 = cache_base[:, None] + f_in2[None, :] * stride_cache_f
            x00 = tl.load(cache_ptr_2d_0, mask=mask_cf, other=0.0)
            x01 = tl.load(cache_ptr_2d_1, mask=mask_cf, other=0.0)
            x02 = tl.load(cache_ptr_2d_2, mask=mask_cf, other=0.0)
        else:
            x_ptr_2d_t0_f0 = x_base[:, None] + t_in0 * stride_x_t + f_in0[None, :] * stride_x_f
            x_ptr_2d_t0_f1 = x_base[:, None] + t_in0 * stride_x_t + f_in1[None, :] * stride_x_f
            x_ptr_2d_t0_f2 = x_base[:, None] + t_in0 * stride_x_t + f_in2[None, :] * stride_x_f
            x00 = tl.load(x_ptr_2d_t0_f0, mask=mask_cf, other=0.0)
            x01 = tl.load(x_ptr_2d_t0_f1, mask=mask_cf, other=0.0)
            x02 = tl.load(x_ptr_2d_t0_f2, mask=mask_cf, other=0.0)

        # Load x for kt=1: (BLOCK_C, BLOCK_F)
        x_ptr_2d_t1_f0 = x_base[:, None] + t_in1 * stride_x_t + f_in0[None, :] * stride_x_f
        x_ptr_2d_t1_f1 = x_base[:, None] + t_in1 * stride_x_t + f_in1[None, :] * stride_x_f
        x_ptr_2d_t1_f2 = x_base[:, None] + t_in1 * stride_x_t + f_in2[None, :] * stride_x_f
        x10 = tl.load(x_ptr_2d_t1_f0, mask=mask_cf, other=0.0)
        x11 = tl.load(x_ptr_2d_t1_f1, mask=mask_cf, other=0.0)
        x12 = tl.load(x_ptr_2d_t1_f2, mask=mask_cf, other=0.0)

        # Load x for kt=2: (BLOCK_C, BLOCK_F)
        x_ptr_2d_t2_f0 = x_base[:, None] + t_in2 * stride_x_t + f_in0[None, :] * stride_x_f
        x_ptr_2d_t2_f1 = x_base[:, None] + t_in2 * stride_x_t + f_in1[None, :] * stride_x_f
        x_ptr_2d_t2_f2 = x_base[:, None] + t_in2 * stride_x_t + f_in2[None, :] * stride_x_f
        x20 = tl.load(x_ptr_2d_t2_f0, mask=mask_cf, other=0.0)
        x21 = tl.load(x_ptr_2d_t2_f1, mask=mask_cf, other=0.0)
        x22 = tl.load(x_ptr_2d_t2_f2, mask=mask_cf, other=0.0)

        # Compute convolution: broadcast weights (BLOCK_C,) -> (BLOCK_C, BLOCK_F)
        acc += w00[:, None] * x00 + w01[:, None] * x01 + w02[:, None] * x02
        acc += w10[:, None] * x10 + w11[:, None] * x11 + w12[:, None] * x12
        acc += w20[:, None] * x20 + w21[:, None] * x21 + w22[:, None] * x22

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        # Store output: (BLOCK_C, BLOCK_F)
        o_ptr_2d = o_base[:, None] + t_out * stride_o_t + idx_f_out[None, :] * stride_o_f
        tl.store(o_ptr_2d, acc, mask=mask_cf)

    # Update cache with last input time frame for next batch
    # IMPORTANT: This must happen AFTER reading cache for t_out=0 above
    # For broadcast mode (in_channels=1), only first output block updates cache
    should_update_cache = True
    if not IS_DEPTHWISE:
        should_update_cache = (tl.program_id(1) == 0)
    
    if should_update_cache:
        tl.debug_barrier()
        last_t = actual_t_len - 1
        if IS_DEPTHWISE:
            cache_mask = mask_c
        else:
            cache_mask = tl.arange(0, BLOCK_C) < 1
        
        # Update cache for all input frequencies at once
        idx_f_cache = tl.arange(0, BLOCK_F)
        mask_f_cache = idx_f_cache < freq_in
        mask_cache_2d = cache_mask[:, None] & mask_f_cache[None, :]
        
        cache_store_ptr = cache_base[:, None] + idx_f_cache[None, :] * stride_cache_f
        x_load_ptr = x_base[:, None] + last_t * stride_x_t + idx_f_cache[None, :] * stride_x_f
        val = tl.load(x_load_ptr, mask=mask_cache_2d, other=0.0)
        tl.store(cache_store_ptr, val, mask=mask_cache_2d)


def causal_conv2d_k3s2_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Causal 2D convolution with kernel=(3,3), stride=(2,2) for continuous batching.

    This is the prefill function for processing variable-length sequences.
    
    Supports two modes:
    - Depthwise: in_channels == out_channels, weight shape (C, 3, 3)
    - Broadcast: in_channels == 1, weight shape (C_out, 3, 3)

    Args:
        x: (C_in, cu_time, F_in) input tensor where cu_time is total time across
           all sequences concatenated. F_in is typically 80 for mel spectrograms.
        weight: (C_out, 3, 3) 2D convolution weights
        bias: (C_out,) optional bias
        conv_state: (num_cache_lines, C_in, 1, F_in) cache storing 1 time frame
        query_start_loc: (batch + 1,) cumulative time lengths prepended by 0
        cache_indices: (batch,) indices into conv_state for each sequence
        has_initial_state: (batch,) whether each sequence has a cached state
        activation: "silu"/"swish" or None
        pad_slot_id: value indicating padded entries in cache_indices
        metadata: optional precomputed metadata for kernel launch
        out: optional pre-allocated output tensor (C_out, cu_time_out, F_out).
             If provided, avoids .item() call for CUDA graph compatibility.

    Returns:
        output: (C_out, cu_time_out, F_out) where:
            cu_time_out = sum((seq_len + 1) // 2 for each seq)
            F_out = (F_in - 3) // 2 + 1
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    # Get dimensions
    in_channels, cu_time, freq_in = x.shape
    out_channels = weight.shape[0]
    assert weight.shape == (out_channels, 3, 3), f"Expected weight shape (C_out, 3, 3), got {weight.shape}"
    
    # Determine mode: depthwise (in==out) or broadcast (in==1)
    is_depthwise = (in_channels == out_channels)
    if not is_depthwise:
        assert in_channels == 1, f"Only depthwise (in==out) or broadcast (in==1) supported, got in={in_channels}, out={out_channels}"

    # Compute output frequency dimension
    freq_out = (freq_in - 3) // 2 + 1
    assert freq_out > 0, f"freq_in={freq_in} too small for kernel=3, stride=2"

    batch = query_start_loc.size(0) - 1

    # Compute output time lengths per sequence
    time_lens = query_start_loc.diff()
    out_time_lens = (time_lens + 1) // 2
    output_start_loc = torch.zeros(
        batch + 1, dtype=query_start_loc.dtype, device=x.device
    )
    output_start_loc[1:] = out_time_lens.cumsum(0)

    # Allocate output (with out_channels) or use pre-allocated buffer
    if out is not None:
        # Use pre-allocated output - avoids .item() for CUDA graph compatibility
        cu_time_out = out.shape[1]
    else:
        # Allocate dynamically - requires CPU sync, not CUDA graph compatible
        cu_time_out = output_start_loc[-1].item()
        out = torch.empty((out_channels, cu_time_out, freq_out), dtype=x.dtype, device=x.device)

    if cu_time_out == 0:
        return out

    # Setup cache indices if not provided
    if cache_indices is None:
        cache_indices = torch.arange(batch, dtype=torch.int32, device=x.device)

    if has_initial_state is None:
        has_initial_state = torch.ones(batch, dtype=torch.bool, device=x.device)

    # Compute program grid
    BLOCK_T = 4  # output time positions per chunk
    BLOCK_C = 64  # channels per block (increased for better parallelism)
    BLOCK_F = 64  # frequency outputs per block (covers all freq_out <= 40)
    
    # Validate frequency constraint
    assert freq_in <= MAX_FREQ_IN, f"freq_in={freq_in} exceeds MAX_FREQ_IN={MAX_FREQ_IN}"
    assert freq_out <= BLOCK_F, f"freq_out={freq_out} exceeds BLOCK_F={BLOCK_F}"

    num_cache_lines = conv_state.size(0)

    if metadata is not None:
        batch_ptr = metadata.batch_ptr
        time_chunk_offset_ptr = metadata.token_chunk_offset_ptr
        # Use pre-computed output_start_loc from metadata if available
        if hasattr(metadata, 'output_start_loc') and metadata.output_start_loc is not None:
            output_start_loc = metadata.output_start_loc
    else:
        # This path uses CPU sync - not compatible with CUDA graph capture
        if out is not None:
            raise RuntimeError(
                "When 'out' is provided for CUDA graph compatibility, "
                "'metadata' must also be provided to avoid CPU sync operations."
            )
        # Compute number of programs needed
        time_lens_cpu = time_lens.cpu().numpy()
        out_time_lens_cpu = (time_lens_cpu + 1) // 2
        num_chunks = (out_time_lens_cpu + BLOCK_T - 1) // BLOCK_T

        total_programs = int(num_chunks.sum())

        # Build batch_ptr and time_chunk_offset_ptr
        batch_list = np.repeat(np.arange(batch), num_chunks)
        offset_list = []
        for n in num_chunks:
            offset_list.extend(range(n))

        batch_ptr = torch.from_numpy(batch_list).to(torch.int32).to(x.device)
        time_chunk_offset_ptr = (
            torch.tensor(offset_list, dtype=torch.int32, device=x.device)
        )

    def grid(META):
        return (
            batch_ptr.size(0),
            triton.cdiv(out_channels, META["BLOCK_C"]),
        )

    _causal_conv2d_k3s2_fwd_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        cache_indices,
        has_initial_state,
        query_start_loc,
        output_start_loc,
        batch_ptr,
        time_chunk_offset_ptr,
        out,
        # Dimensions
        in_channels,
        out_channels,
        freq_in,
        freq_out,
        num_cache_lines,
        # Strides for x
        x.stride(0),
        x.stride(1),
        x.stride(2),
        # Strides for w
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        # Strides for cache
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_state.stride(3),
        # Strides for output
        out.stride(0),
        out.stride(1),
        out.stride(2),
        # Others
        pad_slot_id,
        # Meta
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        USE_PAD_SLOT=pad_slot_id is not None,
        IS_DEPTHWISE=is_depthwise,
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
        BLOCK_F=BLOCK_F,
    )

    return out


def causal_conv2d_k3s2_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Update function for decode mode (processing small number of time frames).
    
    Supports two modes:
    - Depthwise: in_channels == out_channels, weight shape (C, 3, 3)
    - Broadcast: in_channels == 1, weight shape (C_out, 3, 3)

    Args:
        x: Input tensor:
           - (batch, C_in, T, F_in) for fixed-length batched input
           - (C_in, cu_time, F_in) for varlen continuous batching (requires query_start_loc)
        conv_state: (num_cache_lines, C_in, 1, F_in) cache
        weight: (C_out, 3, 3) weights
        bias: (C_out,) optional bias
        activation: "silu"/"swish" or None
        conv_state_indices: (batch,) cache line indices per sequence
        query_start_loc: (batch + 1,) for varlen mode
        pad_slot_id: value indicating padded entries
        out: optional pre-allocated output tensor. If provided, avoids .item()
             call for CUDA graph compatibility.

    Returns:
        output: (batch, C_out, T_out, F_out) or (C_out, cu_time_out, F_out)
    """
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    is_varlen = query_start_loc is not None
    out_channels = weight.shape[0]

    if is_varlen:
        # Varlen mode: x is (C_in, cu_time, F_in)
        in_channels, cu_time, freq_in = x.shape
        batch = query_start_loc.size(0) - 1
        time_len = 0  # Computed per-sequence
        freq_out = (freq_in - 3) // 2 + 1

        # Allocate output or use pre-allocated buffer
        if out is not None:
            # Use pre-allocated output - avoids .item() for CUDA graph compatibility
            cu_time_out = out.shape[1]
        else:
            # Allocate dynamically - requires CPU sync, not CUDA graph compatible
            time_lens = query_start_loc.diff()
            out_time_lens = (time_lens + 1) // 2
            cu_time_out = out_time_lens.sum().item()
            out = torch.empty((out_channels, cu_time_out, freq_out), dtype=x.dtype, device=x.device)

        stride_x_b = 0
        stride_x_c = x.stride(0)
        stride_x_t = x.stride(1)
        stride_x_f = x.stride(2)
        stride_o_b = 0
        stride_o_c = out.stride(0)
        stride_o_t = out.stride(1)
        stride_o_f = out.stride(2)
    else:
        # Fixed length mode: x is (batch, C_in, T, F_in)
        if x.dim() == 3:
            # (C_in, T, F_in) -> (1, C_in, T, F_in)
            x = x.unsqueeze(0)

        batch, in_channels, time_len, freq_in = x.shape
        freq_out = (freq_in - 3) // 2 + 1
        out_time_len = (time_len + 1) // 2

        # Allocate output or use pre-allocated buffer
        if out is None:
            out = torch.empty((batch, out_channels, out_time_len, freq_out), dtype=x.dtype, device=x.device)

        stride_x_b = x.stride(0)
        stride_x_c = x.stride(1)
        stride_x_t = x.stride(2)
        stride_x_f = x.stride(3)
        stride_o_b = out.stride(0)
        stride_o_c = out.stride(1)
        stride_o_t = out.stride(2)
        stride_o_f = out.stride(3)

    # Determine mode
    is_depthwise = (in_channels == out_channels)
    if not is_depthwise:
        assert in_channels == 1, f"Only depthwise (in==out) or broadcast (in==1) supported, got in={in_channels}, out={out_channels}"

    if conv_state_indices is None:
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device)

    assert weight.shape == (out_channels, 3, 3)
    
    # Validate frequency constraint
    assert freq_in <= MAX_FREQ_IN, f"freq_in={freq_in} exceeds MAX_FREQ_IN={MAX_FREQ_IN}"

    num_cache_lines = conv_state.size(0)

    BLOCK_C = 64  # channels per block (increased for better parallelism)
    BLOCK_F = 64  # frequency outputs per block (covers all freq_out <= 40)
    
    assert freq_out <= BLOCK_F, f"freq_out={freq_out} exceeds BLOCK_F={BLOCK_F}"

    def grid(META):
        return (batch, triton.cdiv(out_channels, META["BLOCK_C"]))

    _causal_conv2d_k3s2_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        query_start_loc,
        out,
        # Dimensions
        batch,
        in_channels,
        out_channels,
        time_len,
        freq_in,
        freq_out,
        num_cache_lines,
        # Strides
        stride_x_b,
        stride_x_c,
        stride_x_t,
        stride_x_f,
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_state.stride(3),
        stride_o_b,
        stride_o_c,
        stride_o_t,
        stride_o_f,
        # Others
        pad_slot_id,
        # Meta
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=is_varlen,
        USE_PAD_SLOT=pad_slot_id is not None,
        IS_DEPTHWISE=is_depthwise,
        BLOCK_C=BLOCK_C,
        BLOCK_F=BLOCK_F,
    )

    return out
