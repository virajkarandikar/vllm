# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.conv import stft_cached
from vllm.model_executor.models.fastconformer_preprocessor import (
    MelSpectrogramLayer,
)


def generate_synthetic_audio(
    num_samples: int,
    sample_rate: int = 16000,
    noise_level: float = 0.01,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    variation_seed: int = 0,
) -> torch.Tensor:
    """Generate synthetic audio with time-varying frequency content.

    Creates a mix of:
    - Linear chirp (frequency sweep from 200Hz to 2000Hz)
    - FM-modulated tone (vibrato effect)
    - Pulsed harmonic (amplitude modulated)

    This produces an interesting spectrogram with diagonal lines,
    wavy patterns, and amplitude variations rather than static horizontal lines.

    Args:
        num_samples: Number of audio samples to generate.
        sample_rate: Sample rate in Hz.
        noise_level: Standard deviation of additive Gaussian noise.
        device: Device to place the tensor on.
        dtype: Data type of the tensor.
        variation_seed: Seed to vary audio characteristics. Different seeds
            produce different frequency/modulation parameters for unique audio.

    Returns:
        Audio tensor of shape (num_samples,).
    """
    t = torch.arange(num_samples, device=device, dtype=dtype) / sample_rate
    duration = num_samples / sample_rate

    audio = torch.zeros(num_samples, device=device, dtype=dtype)

    # Vary parameters based on variation_seed
    # Each seed produces a unique combination of frequencies
    seed_offset = variation_seed * 0.1

    # 1. Linear chirp: frequency sweeps from f0 to f1 over time
    f0 = 200.0 + variation_seed * 50.0  # 200, 250, 300, ...
    f1 = 2000.0 + variation_seed * 100.0  # 2000, 2100, 2200, ...
    chirp_rate = (f1 - f0) / duration
    phase_chirp = 2 * torch.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    audio += 0.4 * torch.sin(phase_chirp)

    # 2. FM-modulated tone (vibrato): carrier with sinusoidal frequency modulation
    carrier_freq = 800.0 + variation_seed * 75.0  # 800, 875, 950, ...
    mod_freq = 5.0 + variation_seed * 1.5  # 5.0, 6.5, 8.0, ...
    mod_depth = 100.0 + variation_seed * 20.0  # 100, 120, 140, ...
    phase_fm = 2 * torch.pi * (
        carrier_freq * t + (mod_depth / mod_freq) * torch.sin(2 * torch.pi * mod_freq * t)
    )
    audio += 0.3 * torch.sin(phase_fm)

    # 3. Amplitude-modulated harmonic (pulsing tone)
    pulse_freq = 600.0 + variation_seed * 80.0  # 600, 680, 760, ...
    am_freq = 3.0 + variation_seed * 0.7  # 3.0, 3.7, 4.4, ...
    envelope = 0.5 * (1 + torch.sin(2 * torch.pi * am_freq * t + seed_offset))
    audio += 0.3 * envelope * torch.sin(2 * torch.pi * pulse_freq * t)

    # Normalize to prevent clipping
    audio = audio / audio.abs().max()

    # Add noise (use variation_seed for reproducible but different noise)
    if noise_level > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(42 + variation_seed)
        noise = torch.randn(
            num_samples, device=device, dtype=dtype, generator=generator
        ) * noise_level
        audio = audio + noise

    return audio


def compute_stft_reference(
    audio_padded: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window_length: int,
) -> torch.Tensor:
    """Compute STFT power spectrum using torch.stft (reference implementation).

    Args:
        audio_padded: Padded audio tensor of shape (1, samples) or (samples,).
        n_fft: FFT size.
        hop_length: Hop length between frames.
        window_length: Window length for STFT.

    Returns:
        Power spectrum of shape (num_frames, freq_bins).
    """
    if audio_padded.dim() == 1:
        audio_padded = audio_padded.unsqueeze(0)

    window = torch.hann_window(
        window_length,
        periodic=False,
        device=audio_padded.device,
        dtype=audio_padded.dtype,
    )
    spec = torch.stft(
        audio_padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        center=False,
        window=window,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1))
    spec = spec.pow(2.0)  # Power spectrum
    # Output shape: (1, freq_bins, num_frames) -> transpose to (num_frames, freq_bins)
    return spec[0].T


def compute_stft_kernel(
    audio: torch.Tensor,
    wcos: torch.Tensor,
    wsin: torch.Tensor,
    n_fft: int,
    hop_length: int,
    prompt_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute STFT using the custom kernel with streaming simulation.

    Args:
        audio: Raw audio tensor of shape (samples,).
        wcos: Real (cosine) basis from MelSpectrogramLayer.
        wsin: Imaginary (sine) basis from MelSpectrogramLayer.
        n_fft: FFT size.
        hop_length: Hop length between frames.
        prompt_len: Number of frames to process in the first chunk (prompt).
        dtype: Data type.

    Returns:
        Power spectrum of shape (num_frames, freq_bins).
    """
    freq_bins = n_fft // 2 + 1

    # Prepare cache
    # NOTE: in fastconformer preprocessor we use padded cache,
    # so here the cache size is 130000 instead of nfft - hop_len
    cache = torch.zeros(10, 130000, device="cuda", dtype=dtype)
    cache_indices = torch.zeros(1, dtype=torch.int64, device="cuda")

    # Run inference frame by frame (simulating streaming)
    i = 0
    spec_lst = []
    num_frames = audio.shape[0] // hop_length

    while i < num_frames:
        chunk_size = prompt_len if i == 0 else 1
        chunk_size = min(chunk_size, num_frames - i)
        chunk = audio[i * hop_length : (i + chunk_size) * hop_length]

        out = torch.empty(chunk_size, freq_bins, device="cuda", dtype=dtype)
        query_start_loc = torch.tensor(
            [0, chunk_size],
            dtype=torch.int32,
            device="cuda",
        )

        stft_cached(
            chunk,
            wcos,
            wsin,
            out,
            cache,
            query_start_loc,
            cache_indices,
            n_fft=n_fft,
            hop_length=hop_length,
            time_factor=hop_length,
            output_divisor=hop_length,
            metadata=None,
        )
        spec_lst.append(out.clone())
        i += chunk_size

    # Concatenate all frames
    return torch.cat(spec_lst, dim=0)


def compute_stft_kernel_batched(
    audio_list: list[torch.Tensor],
    wcos: torch.Tensor,
    wsin: torch.Tensor,
    n_fft: int,
    hop_length: int,
    prompt_len: int,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Compute STFT using the custom kernel for N sequences (batched).

    Simulates streaming inference where multiple sequences are processed
    together in each kernel call.

    Args:
        audio_list: List of N audio tensors, each of shape (samples_i,).
        wcos: Real (cosine) basis from MelSpectrogramLayer.
        wsin: Imaginary (sine) basis from MelSpectrogramLayer.
        n_fft: FFT size.
        hop_length: Hop length between frames.
        prompt_len: Number of frames to process in the first chunk (prompt).
        dtype: Data type.

    Returns:
        List of N power spectrum tensors, each of shape (num_frames_i, freq_bins).
    """
    freq_bins = n_fft // 2 + 1
    n_seqs = len(audio_list)

    # Prepare cache - one entry per sequence
    cache = torch.zeros(n_seqs, 130000, device="cuda", dtype=dtype)
    cache_indices = torch.arange(n_seqs, dtype=torch.int64, device="cuda")

    # Track frame position for each sequence
    num_frames_list = [audio.shape[0] // hop_length for audio in audio_list]
    frame_positions = [0] * n_seqs  # Current frame position for each sequence

    # Collect output for each sequence
    spec_lists: list[list[torch.Tensor]] = [[] for _ in range(n_seqs)]

    # Run inference frame by frame (simulating streaming)
    # Continue until all sequences are done
    while any(pos < num_frames_list[i] for i, pos in enumerate(frame_positions)):
        # Determine chunk size for each sequence
        chunk_sizes = []
        chunks = []
        active_seq_indices = []

        for seq_idx in range(n_seqs):
            pos = frame_positions[seq_idx]
            remaining = num_frames_list[seq_idx] - pos
            if remaining <= 0:
                continue  # This sequence is done

            chunk_size = prompt_len if pos == 0 else 1
            chunk_size = min(chunk_size, remaining)
            chunk_sizes.append(chunk_size)
            active_seq_indices.append(seq_idx)

            # Extract chunk for this sequence
            audio = audio_list[seq_idx]
            chunk = audio[pos * hop_length : (pos + chunk_size) * hop_length]
            chunks.append(chunk)

        if not chunks:
            break

        # Concatenate all chunks
        audio_packed = torch.cat(chunks, dim=0)

        # Build query_start_loc: N+1 elements with cumulative frame counts
        # query_start_loc[i] = sum of chunk_sizes[0:i]
        query_start_loc_list = [0]
        cumsum = 0
        for cs in chunk_sizes:
            cumsum += cs
            query_start_loc_list.append(cumsum)
        query_start_loc = torch.tensor(
            query_start_loc_list,
            dtype=torch.int32,
            device="cuda",
        )

        total_frames = sum(chunk_sizes)
        out = torch.empty(total_frames, freq_bins, device="cuda", dtype=dtype)

        # Build cache_indices for active sequences only
        active_cache_indices = torch.tensor(
            active_seq_indices, dtype=torch.int64, device="cuda"
        )

        stft_cached(
            audio_packed,
            wcos,
            wsin,
            out,
            cache,
            query_start_loc,
            active_cache_indices,
            n_fft=n_fft,
            hop_length=hop_length,
            time_factor=hop_length,
            output_divisor=hop_length,
            metadata=None,
        )

        # Split output and assign to each sequence
        offset = 0
        for i, seq_idx in enumerate(active_seq_indices):
            cs = chunk_sizes[i]
            spec_lists[seq_idx].append(out[offset : offset + cs].clone())
            offset += cs
            frame_positions[seq_idx] += cs

    # Concatenate frames for each sequence
    return [torch.cat(spec_list, dim=0) for spec_list in spec_lists]


@pytest.mark.parametrize("prompt_len", [1, 20, 100])
@pytest.mark.parametrize("num_frames", [9, 43, 67])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_stft_cached_vs_torch(prompt_len: int, num_frames: int, dtype: torch.dtype):
    """Test that stft_cached kernel matches torch.stft output.

    Args:
        prompt_len: Number of frames to process in the first chunk.
        dtype: Data type for computation.
    """
    device = "cuda"

    # STFT parameters (matching FastConformer defaults)
    n_fft = 512
    hop_length = 160
    window_length = 400
    n_filt = 80
    sample_rate = 16000
    mag_power = 2.0

    # Generate synthetic audio
    # Use a length that gives a reasonable number of frames
    num_samples = num_frames * hop_length
    audio = generate_synthetic_audio(
        num_samples=num_samples,
        sample_rate=sample_rate,
        device=device,
        dtype=dtype,
    )

    # Pad audio for reference computation (as done in notebook)
    pad = n_fft - hop_length
    audio_padded = torch.nn.functional.pad(audio.unsqueeze(0), (pad, 0))

    # Compute reference using torch.stft
    spec_ref = compute_stft_reference(
        audio_padded,
        n_fft=n_fft,
        hop_length=hop_length,
        window_length=window_length,
    )

    # Create MelSpectrogramLayer to get wcos/wsin basis
    melspec_layer = MelSpectrogramLayer(
        time_factor=hop_length,
        prefix="",
        cache_config=None,
        dtype=dtype,
        window_length=window_length,
        hop_length=hop_length,
        n_fft=n_fft,
        mag_power=mag_power,
        n_filt=n_filt,
        sample_rate=sample_rate,
    )
    melspec_layer.init_stft_basis()
    wcos = melspec_layer.wcos.data.clone().to(device)
    wsin = melspec_layer.wsin.data.clone().to(device)

    # Compute using kernel
    spec_kernel = compute_stft_kernel(
        audio,
        wcos,
        wsin,
        n_fft=n_fft,
        hop_length=hop_length,
        prompt_len=prompt_len,
        dtype=dtype,
    )

    # Compare results
    # Allow for small numerical differences due to different computation order
    # TODO: precision is 1e-2 because amplitude of ampl spectrum is ~2000.
    rtol = 1e-2
    atol = 1e-2

    assert spec_kernel.shape == spec_ref.shape, (
        f"Shape mismatch: kernel {spec_kernel.shape} vs ref {spec_ref.shape}"
    )

    max_diff = (spec_kernel - spec_ref).abs().max().item()
    mean_diff = (spec_kernel - spec_ref).abs().mean().item()

    assert torch.allclose(spec_kernel, spec_ref, rtol=rtol, atol=atol), (
        f"STFT kernel output doesn't match torch.stft reference. "
        f"prompt_len={prompt_len}, max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
    )


@pytest.mark.parametrize("prompt_len", [1, 20, 100])
@pytest.mark.parametrize("n_seqs", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_stft_cached_batched_vs_torch(
    prompt_len: int, n_seqs: int, dtype: torch.dtype
):
    """Test that batched stft_cached kernel matches torch.stft for N sequences.

    This test validates that when processing N sequences packed together:
    - Each sequence gets correct STFT output matching torch.stft reference
    - Cache is properly maintained per-sequence
    - query_start_loc correctly separates the sequences

    Args:
        prompt_len: Number of frames to process in the first chunk.
        n_seqs: Number of sequences to process together.
        dtype: Data type for computation.
    """
    device = "cuda"

    # STFT parameters (matching FastConformer defaults)
    n_fft = 512
    hop_length = 160
    window_length = 400
    n_filt = 80
    sample_rate = 16000
    mag_power = 2.0

    # Generate synthetic audio for each sequence with varying lengths
    # Use different frame counts to test variable-length sequences
    frame_counts = [9 + i * 7 for i in range(n_seqs)]  # e.g., [9, 16, 23, ...]

    audio_list = []
    audio_padded_list = []
    pad = n_fft - hop_length

    for seq_idx, num_frames in enumerate(frame_counts):
        num_samples = num_frames * hop_length
        audio = generate_synthetic_audio(
            num_samples=num_samples,
            sample_rate=sample_rate,
            device=device,
            dtype=dtype,
            variation_seed=seq_idx,  # Different audio for each sequence
        )
        audio_list.append(audio)
        # Pad for reference computation
        audio_padded = torch.nn.functional.pad(audio.unsqueeze(0), (pad, 0))
        audio_padded_list.append(audio_padded)

    # Compute reference for each sequence using torch.stft
    spec_ref_list = [
        compute_stft_reference(
            audio_padded,
            n_fft=n_fft,
            hop_length=hop_length,
            window_length=window_length,
        )
        for audio_padded in audio_padded_list
    ]

    # Create MelSpectrogramLayer to get wcos/wsin basis
    melspec_layer = MelSpectrogramLayer(
        time_factor=hop_length,
        prefix="",
        cache_config=None,
        dtype=dtype,
        window_length=window_length,
        hop_length=hop_length,
        n_fft=n_fft,
        mag_power=mag_power,
        n_filt=n_filt,
        sample_rate=sample_rate,
    )
    melspec_layer.init_stft_basis()
    wcos = melspec_layer.wcos.data.clone().to(device)
    wsin = melspec_layer.wsin.data.clone().to(device)

    # Compute using batched kernel
    spec_kernel_list = compute_stft_kernel_batched(
        audio_list,
        wcos,
        wsin,
        n_fft=n_fft,
        hop_length=hop_length,
        prompt_len=prompt_len,
        dtype=dtype,
    )

    # Compare results for each sequence
    rtol = 1e-2
    atol = 1e-2

    for seq_idx, (spec_kernel, spec_ref) in enumerate(
        zip(spec_kernel_list, spec_ref_list)
    ):
        assert spec_kernel.shape == spec_ref.shape, (
            f"Seq {seq_idx}: Shape mismatch: "
            f"kernel {spec_kernel.shape} vs ref {spec_ref.shape}"
        )

        max_diff = (spec_kernel - spec_ref).abs().max().item()
        mean_diff = (spec_kernel - spec_ref).abs().mean().item()

        assert torch.allclose(spec_kernel, spec_ref, rtol=rtol, atol=atol), (
            f"Seq {seq_idx}: STFT kernel output doesn't match torch.stft reference. "
            f"prompt_len={prompt_len}, n_seqs={n_seqs}, "
            f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
        )
