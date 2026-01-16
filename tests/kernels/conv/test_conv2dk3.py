# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for depthwise_strided_conv2d_cached kernel.

This kernel implements a 3x3 depthwise strided 2D convolution with stride=2
in both time and frequency dimensions, with cache support for causal streaming.
"""

import pytest
import torch

from vllm.model_executor.layers.conv import depthwise_strided_conv2d_cached


def generate_random_input(
    time_len: int,
    freq: int,
    channels: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random input tensor for conv2d testing.

    Args:
        time_len: Number of time steps.
        freq: Number of frequency bins.
        channels: Number of channels.
        device: Device to place the tensor on.
        dtype: Data type of the tensor.
        seed: Random seed for reproducibility.

    Returns:
        Random tensor of shape (time_len, freq, channels).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        time_len, freq, channels, device=device, dtype=dtype, generator=generator
    )


def generate_random_weights(
    channels: int,
    kernel_size: int = 3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random conv2d weights and bias.

    Args:
        channels: Number of channels (depthwise conv).
        kernel_size: Kernel size (default 3).
        device: Device to place tensors on.
        dtype: Data type.
        seed: Random seed.

    Returns:
        Tuple of (weight, bias) tensors.
        weight: (kernel_size, kernel_size, channels)
        bias: (channels,)
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    weight = torch.randn(
        kernel_size, kernel_size, channels,
        device=device, dtype=dtype, generator=generator
    )
    bias = torch.randn(channels, device=device, dtype=dtype, generator=generator)
    return weight, bias


def compute_conv2d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute reference depthwise strided conv2d using torch.nn.functional.conv2d.

    The kernel stores weights in a flipped format compared to torch.conv2d.
    This matches the weight transformation in load_weights:
        nemo_weight.squeeze(1).permute(1, 2, 0).flip(1).flip(0)

    To use torch.conv2d, we reverse the transformation:
        kernel_weight.flip(0).flip(1).permute(2, 0, 1).unsqueeze(1)

    Args:
        x: Input tensor (T, F, C).
        weight: Kernel weights (3, 3, C) in the custom kernel's format.
        bias: Bias (C,).

    Returns:
        Output tensor (T//2, F//2, C).
    """
    T, F, C = x.shape

    # Reshape input: (T, F, C) -> (1, C, T, F) for conv2d format (N, C, H, W)
    x_nchw = x.permute(2, 0, 1).unsqueeze(0)

    # Transform weights from kernel format to conv2d format:
    # - Flip in both dimensions (kernel uses correlation-like indexing)
    # - Reshape from (3, 3, C) to (C, 1, 3, 3) for depthwise conv
    w_flipped = weight.flip(0).flip(1)  # (3, 3, C)
    w_conv = w_flipped.permute(2, 0, 1).unsqueeze(1)  # (C, 1, 3, 3)

    # Apply depthwise conv2d with stride=2, padding=(1, 1)
    out = torch.nn.functional.conv2d(
        x_nchw,
        w_conv,
        bias=bias,
        stride=2,
        padding=(1, 1),
        groups=C,
    )

    # Reshape output: (1, C, T_out, F_out) -> (T_out, F_out, C)
    return out.squeeze(0).permute(1, 2, 0)


def compute_conv2d_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    freq: int,
    channels: int,
    prompt_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute conv2d using the custom kernel with streaming simulation.

    Args:
        x: Input tensor (T, F, C).
        weight: Kernel weights (3, 3, C).
        bias: Bias (C,).
        freq: Frequency dimension size.
        channels: Number of channels.
        prompt_len: Number of output frames to process in the first chunk.
        dtype: Data type.

    Returns:
        Output tensor (T//2, F//2, C).
    """
    T = x.shape[0]
    T_out = T // 2
    F_out = freq // 2

    # Prepare cache - shape: (num_seqs, freq, channels)
    # The cache stores the last odd time step values per frequency
    # NOTE: using padded cache as in fastconformer preprocessor
    cache = torch.zeros(10, 256, channels, device="cuda", dtype=dtype)
    cache_indices = torch.zeros(1, dtype=torch.int64, device="cuda")

    # Simulate streaming: process in chunks
    i = 0
    out_lst = []

    while i < T_out:
        chunk_out_size = prompt_len if i == 0 else 1
        chunk_out_size = min(chunk_out_size, T_out - i)
        chunk_in_size = chunk_out_size * 2  # stride=2

        # Extract input chunk
        chunk_in = x[i * 2: i * 2 + chunk_in_size]

        out = torch.empty(chunk_out_size, F_out, channels, device="cuda", dtype=dtype)
        query_start_loc = torch.tensor(
            [0, chunk_out_size],
            dtype=torch.int32,
            device="cuda",
        )

        # has_initial_state: True after first chunk (cache is valid)
        has_initial_state = torch.tensor(
            [i > 0], dtype=torch.bool, device="cuda"
        )

        depthwise_strided_conv2d_cached(
            chunk_in,
            weight,
            bias,
            out,
            cache,
            query_start_loc,
            cache_indices,
            has_initial_state,
            time_factor=1,
            output_divisor=1,
            metadata=None,
        )
        out_lst.append(out.clone())
        i += chunk_out_size

    return torch.cat(out_lst, dim=0)


def compute_conv2d_kernel_batched(
    x_list: list[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    freq: int,
    channels: int,
    prompt_len: int,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Compute conv2d using the custom kernel for N sequences (batched).

    Simulates streaming inference where multiple sequences are processed
    together in each kernel call.

    Args:
        x_list: List of N input tensors, each of shape (T_i, F, C).
        weight: Kernel weights (3, 3, C).
        bias: Bias (C,).
        freq: Frequency dimension size.
        channels: Number of channels.
        prompt_len: Number of output frames to process in the first chunk.
        dtype: Data type.

    Returns:
        List of N output tensors, each of shape (T_i//2, F//2, C).
    """
    n_seqs = len(x_list)
    F_out = freq // 2

    # Prepare cache - one entry per sequence
    # NOTE: using padded cache as in fastconformer preprocessor
    cache = torch.zeros(n_seqs, 256, channels, device="cuda", dtype=dtype)

    # Track output frame position for each sequence
    T_out_list = [x.shape[0] // 2 for x in x_list]
    frame_positions = [0] * n_seqs  # Current output frame position

    # Collect outputs for each sequence
    out_lists: list[list[torch.Tensor]] = [[] for _ in range(n_seqs)]

    # Simulate streaming - continue until all sequences are done
    while any(pos < T_out_list[i] for i, pos in enumerate(frame_positions)):
        chunk_out_sizes = []
        chunks = []
        active_seq_indices = []

        for seq_idx in range(n_seqs):
            pos = frame_positions[seq_idx]
            remaining = T_out_list[seq_idx] - pos
            if remaining <= 0:
                continue  # This sequence is done

            chunk_out_size = prompt_len if pos == 0 else 1
            chunk_out_size = min(chunk_out_size, remaining)
            chunk_out_sizes.append(chunk_out_size)
            active_seq_indices.append(seq_idx)

            # Extract input chunk (stride=2)
            chunk_in_size = chunk_out_size * 2
            x = x_list[seq_idx]
            chunk = x[pos * 2: pos * 2 + chunk_in_size]
            chunks.append(chunk)

        if not chunks:
            break

        # Pack all chunks together
        x_packed = torch.cat(chunks, dim=0)

        # Build query_start_loc from cumulative INPUT frame counts
        # The kernel expects: input_time = query_start_loc * time_factor
        # With stride=2: output_time = input_time // output_divisor
        query_start_loc_list = [0]
        cumsum = 0
        for cs in chunk_out_sizes:
            chunk_in_size = cs * 2  # stride=2
            cumsum += chunk_in_size
            query_start_loc_list.append(cumsum)
        query_start_loc = torch.tensor(
            query_start_loc_list,
            dtype=torch.int32,
            device="cuda",
        )

        total_out_frames = sum(chunk_out_sizes)
        out = torch.empty(total_out_frames, F_out, channels, device="cuda", dtype=dtype)

        # Build cache_indices and has_initial_state for active sequences
        active_cache_indices = torch.tensor(
            active_seq_indices, dtype=torch.int64, device="cuda"
        )
        has_initial_state = torch.tensor(
            [frame_positions[idx] > 0 for idx in active_seq_indices],
            dtype=torch.bool, device="cuda"
        )

        depthwise_strided_conv2d_cached(
            x_packed,
            weight,
            bias,
            out,
            cache,
            query_start_loc,
            active_cache_indices,
            has_initial_state,
            time_factor=1,
            output_divisor=2,  # stride=2: output positions = input positions // 2
            metadata=None,
        )

        # Split output and assign to each sequence
        offset = 0
        for i, seq_idx in enumerate(active_seq_indices):
            cs = chunk_out_sizes[i]
            out_lists[seq_idx].append(out[offset: offset + cs].clone())
            offset += cs
            frame_positions[seq_idx] += cs

    return [torch.cat(out_list, dim=0) for out_list in out_lists]


@pytest.mark.parametrize("prompt_len", [1, 4, 20])
@pytest.mark.parametrize("time_len", [8, 22, 64])
@pytest.mark.parametrize("freq", [16, 44, 88])
@pytest.mark.parametrize("channels", [32, 256])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv2d_cached_vs_reference(
    prompt_len: int,
    time_len: int,
    freq: int,
    channels: int,
    dtype: torch.dtype,
):
    """Test that depthwise_strided_conv2d_cached matches reference implementation.

    Args:
        prompt_len: Number of output frames to process in the first chunk.
        time_len: Input time dimension (must be even).
        freq: Frequency dimension (must be even).
        channels: Number of channels.
        dtype: Data type for computation.
    """
    device = "cuda"

    # Ensure even dimensions for stride=2
    time_len = (time_len // 2) * 2
    freq = (freq // 2) * 2

    # Generate random input and weights
    x = generate_random_input(
        time_len=time_len,
        freq=freq,
        channels=channels,
        device=device,
        dtype=dtype,
    )
    weight, bias = generate_random_weights(
        channels=channels,
        device=device,
        dtype=dtype,
    )

    # Compute reference
    out_ref = compute_conv2d_reference(x, weight, bias)

    # Compute using kernel with streaming
    out_kernel = compute_conv2d_kernel(
        x,
        weight,
        bias,
        freq=freq,
        channels=channels,
        prompt_len=prompt_len,
        dtype=dtype,
    )

    # Compare results
    rtol = 1e-4
    atol = 1e-4

    assert out_kernel.shape == out_ref.shape, (
        f"Shape mismatch: kernel {out_kernel.shape} vs ref {out_ref.shape}"
    )

    max_diff = (out_kernel - out_ref).abs().max().item()
    mean_diff = (out_kernel - out_ref).abs().mean().item()

    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"Conv2d kernel output doesn't match reference. "
        f"prompt_len={prompt_len}, time_len={time_len}, freq={freq}, channels={channels}, "
        f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
    )


@pytest.mark.parametrize("prompt_len", [1, 4, 20])
@pytest.mark.parametrize("n_seqs", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv2d_cached_batched_vs_reference(
    prompt_len: int,
    n_seqs: int,
    dtype: torch.dtype,
):
    """Test batched depthwise_strided_conv2d_cached matches reference for N sequences.

    This test validates that when processing N sequences packed together:
    - Each sequence gets correct output matching reference
    - Cache is properly maintained per-sequence
    - query_start_loc correctly separates the sequences

    Args:
        prompt_len: Number of output frames to process in the first chunk.
        n_seqs: Number of sequences to process together.
        dtype: Data type for computation.
    """
    device = "cuda"
    freq = 44
    channels = 256

    # Generate inputs with varying time lengths
    time_lens = [8 + i * 6 for i in range(n_seqs)]  # e.g., [8, 14, 20, ...]
    # Ensure even
    time_lens = [(t // 2) * 2 for t in time_lens]

    x_list = [
        generate_random_input(
            time_len=t,
            freq=freq,
            channels=channels,
            device=device,
            dtype=dtype,
            seed=42 + i,
        )
        for i, t in enumerate(time_lens)
    ]

    # Use same weights for all sequences
    weight, bias = generate_random_weights(
        channels=channels,
        device=device,
        dtype=dtype,
    )

    # Compute reference for each sequence
    out_ref_list = [compute_conv2d_reference(x, weight, bias) for x in x_list]

    # Compute using batched kernel
    out_kernel_list = compute_conv2d_kernel_batched(
        x_list,
        weight,
        bias,
        freq=freq,
        channels=channels,
        prompt_len=prompt_len,
        dtype=dtype,
    )

    # Compare results for each sequence
    rtol = 1e-4
    atol = 1e-4

    for seq_idx, (out_kernel, out_ref) in enumerate(
        zip(out_kernel_list, out_ref_list)
    ):
        assert out_kernel.shape == out_ref.shape, (
            f"Seq {seq_idx}: Shape mismatch: "
            f"kernel {out_kernel.shape} vs ref {out_ref.shape}"
        )

        max_diff = (out_kernel - out_ref).abs().max().item()
        mean_diff = (out_kernel - out_ref).abs().mean().item()

        assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
            f"Seq {seq_idx}: Conv2d kernel output doesn't match reference. "
            f"prompt_len={prompt_len}, n_seqs={n_seqs}, "
            f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
        )


@pytest.mark.parametrize("time_len", [8, 20, 50])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv2d_streaming_consistency(
    time_len: int,
    dtype: torch.dtype,
):
    """Test that different chunk sizes produce the same result.

    This verifies the cache mechanism works correctly by comparing:
    - Full prefill (all frames at once)
    - Frame-by-frame decode (chunk_size=1)

    Args:
        time_len: Input time dimension.
        dtype: Data type for computation.
    """
    device = "cuda"
    freq = 44
    channels = 256

    # Ensure even
    time_len = (time_len // 2) * 2

    x = generate_random_input(
        time_len=time_len,
        freq=freq,
        channels=channels,
        device=device,
        dtype=dtype,
    )
    weight, bias = generate_random_weights(
        channels=channels,
        device=device,
        dtype=dtype,
    )

    T_out = time_len // 2

    # Full prefill: process all at once
    out_full = compute_conv2d_kernel(
        x, weight, bias,
        freq=freq, channels=channels,
        prompt_len=T_out,  # All frames in one chunk
        dtype=dtype,
    )

    # Frame-by-frame: process 1 output frame at a time
    out_streaming = compute_conv2d_kernel(
        x, weight, bias,
        freq=freq, channels=channels,
        prompt_len=1,  # One frame at a time
        dtype=dtype,
    )

    rtol = 1e-5
    atol = 1e-5

    max_diff = (out_full - out_streaming).abs().max().item()
    assert torch.allclose(out_full, out_streaming, rtol=rtol, atol=atol), (
        f"Full prefill and streaming outputs differ. max_diff={max_diff:.6e}"
    )

