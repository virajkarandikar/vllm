# SPDX-License-Identifier: Apache-2.0
"""
Implements fastconformer preprocessing module: mel-spectrogram extraction and subsampling.
We don't implement pre-emphasis here, this has to be done externally.
There are differences from NeMo implementation to ensure proper streaming support:
    * STFT is non-centered
    * Subsampling in NeMo is uneven, it pads frequency with (2, 1). That results
    in 80 frequency bins been downsampled to 11 instead of 10.
    We correctly pad (1, 0), but to ensure same dimension, we pre-pad 80 to 88.

Overall downsampling is x1280 (160 hop-length and x8 subsampling), so audio can be fed
in chunks of 1280 samples. Similar to FastConformer, multi-frame prefill stage does not
work with CUDA graphs enabled.
"""

from typing import Optional, Iterable

import librosa
import torch
import torch.nn as nn
import numpy as np
import math

from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.fastconformer_conv import (
    FastConformerConvBackend,
    FastConformerConvMetadata,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.custom_op import CustomOp
from vllm.v1.kv_cache_interface import KVCacheSpec, FastConformerConvSpec
from vllm.model_executor.layers.conv import depthwise_strided_conv2d_cached, stft_cached
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.utils import direct_register_custom_op
from vllm.compilation.decorators import support_torch_compile


LOG_ZERO_GUARD_VALUE = 5.960464477539063e-08
SAMPLE_RATE = 16000
FASTCONFORMER_CACHE_PAGE_SIZE = 256 * 1024


@CustomOp.register("mel_spec_layer")
class MelSpectrogramLayer(CustomOp, AttentionLayerBase):
    """
    Extracts mel spectrogram from the audio

    Processes (Samples,) -> (Frames, Frequencies)
    For cached inference feed in chuks of `hop_length * N`
    """

    def __init__(
        self,
        time_factor: int,
        prefix: str,
        cache_config: CacheConfig,
        dtype: torch.dtype,
        window_length=400,
        hop_length=160,
        n_fft=512,
        mag_power=2.0,
        n_filt=128,
        sample_rate=SAMPLE_RATE,
        **kwargs,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag_power = float(mag_power)
        self.n_filt = n_filt
        self.freq_bins = n_fft // 2 + 1
        self.cache_len = n_fft - window_length
        self.window_length = window_length
        self.sample_rate = sample_rate

        # WARNING! stft requires 1d cache. cache pages should have the same size across layers.
        # we have to use a cache that corresponds to the rest of FastConformer layers.
        self.pad_cache_len = FASTCONFORMER_CACHE_PAGE_SIZE
        assert self.pad_cache_len >= self.cache_len
        self.log_zero_guard_value = LOG_ZERO_GUARD_VALUE

        # definitions used by vllm layer with cache
        self.time_factor = time_factor
        self.cache_config = cache_config
        self.kv_cache = [torch.tensor([])]
        self.dtype = dtype
        self.prefix = prefix
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

        # STFT basis functions and mel filterbank - initialized as parameters
        # so they are properly moved to CUDA during model loading.
        # Actual values are computed in load_weights().
        self.wsin = nn.Parameter(
            torch.zeros(self.freq_bins, n_fft, dtype=dtype),
            requires_grad=False,
        )
        self.wcos = nn.Parameter(
            torch.zeros(self.freq_bins, n_fft, dtype=dtype),
            requires_grad=False,
        )
        self.fb = nn.Parameter(
            torch.zeros(self.freq_bins, n_filt, dtype=dtype),
            requires_grad=False,
        )

    def init_stft_basis(self):
        """Compute STFT basis functions and mel filterbank.

        Called during load_weights() after model is on device.
        """
        # Prepare windowed basis functions
        window = np.hanning(self.window_length).astype(np.float32)
        pad_left = (self.n_fft - self.window_length) // 2
        pad_right = self.n_fft - self.window_length - pad_left
        window_centered = np.pad(window, (pad_left, pad_right), mode="constant")

        s = np.arange(0, self.n_fft, dtype=np.float32)
        wsin = np.zeros((self.freq_bins, self.n_fft), dtype=np.float32)
        wcos = np.zeros((self.freq_bins, self.n_fft), dtype=np.float32)
        for k in range(self.freq_bins):
            wsin[k, :] = np.sin(2 * np.pi * k * s / self.n_fft) * window_centered
            wcos[k, :] = np.cos(2 * np.pi * k * s / self.n_fft) * window_centered

        self.wsin.data.copy_(torch.from_numpy(wsin).to(dtype=self.dtype))
        self.wcos.data.copy_(torch.from_numpy(wcos).to(dtype=self.dtype))

        # Mel filterbank
        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_filt,
                fmin=0,
                fmax=self.sample_rate / 2,
                norm="slaney",
            ),
            dtype=self.dtype,
        ).transpose(
            0, 1
        )  # freq_bins x n_filt
        self.fb.data.copy_(filterbanks)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.mel_spec_layer(
            audio,
            self.prefix,
        )

    def forward_cuda(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (Samples,) packed audio to extract melspec from.
            For proper streaming, the audio should be divisible by hop_length.
        Returns:
            melspec: (Frames, Frequencies), where Frames = Samples // hop_length
        """
        assert audio.dim() == 1, "audio should be 1D tensor"
        assert (
            audio.shape[0] >= self.hop_length
        ), "audio should be at least one hop_length long"

        seq_len = audio.shape[0]
        out_seq_len = seq_len // self.hop_length

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata

        if attn_meta_all is None:
            # profile run, return zeros with correct output shape
            return torch.zeros(
                (out_seq_len, self.n_filt),
                dtype=audio.dtype,
                device=audio.device,
            )

        x = audio.contiguous()
        out = torch.empty(
            (out_seq_len, self.n_fft // 2 + 1), device=x.device, dtype=x.dtype
        )
        attn_metadata: FastConformerConvMetadata = attn_meta_all[self.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        # Cache store: (num_blocks, samples)
        # Only self.cache_len is used, rest is ignored
        store = self.kv_cache[fctx.virtual_engine]
        stft_cached(
            x,
            self.wcos,  # real
            self.wsin,  # imag
            out,
            store,
            attn_metadata.query_start_loc,
            page_indices,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            # time_factor converts query_start_loc to sample positions
            time_factor=self.time_factor,
            # output_divisor converts samples to frames (= hop_length)
            output_divisor=self.hop_length,
            metadata=attn_metadata,
        )

        # compute mel spec
        x = torch.matmul(out, self.fb)
        x = torch.log(x + self.log_zero_guard_value)

        return x

    def get_attn_backend(self) -> AttentionBackend:
        # Use FastConformerConvBackend - STFT kernel handles time_factor and
        # output_divisor internally, no need for pre-computed output positions
        return FastConformerConvBackend

    @property
    def output_elements_per_token(self) -> int:
        """Number of output frames produced per input token."""
        return self.time_factor // self.hop_length

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            block_size=1,
            shape=(self.pad_cache_len,),
            dtype=self.dtype,
        )


def mel_spec_fwd(
    audio: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.forward_cuda(audio=audio)


def mel_spec_fwd_fake(
    audio: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:

    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    # output shape: (frames_num, freq_bins)
    samples_num = audio.shape[0]
    frames_num = samples_num // self.hop_length
    return torch.zeros(
        (frames_num, self.n_filt),
        dtype=audio.dtype,
        device=audio.device,
    )


direct_register_custom_op(
    op_name="mel_spec_layer",
    op_func=mel_spec_fwd,
    fake_impl=mel_spec_fwd_fake,
)


KERNEL_SIZE = 3
STRIDE = 2


@CustomOp.register("conv2d_layer")
class Conv2dLayer(CustomOp, AttentionLayerBase):
    """
    Cached depthwise strided 2D convolution layer with kernel=3, stride=2.

    Processes (T, Freq, Channels) -> (T/2, Freq/2, Channels).
    Maintains cache for causal streaming over time dimension.
    """

    def __init__(
        self,
        time_factor: int,
        prefix: str,
        cache_config: CacheConfig,
        dtype: torch.dtype,
        channels: int,
        freq: int,
        kernel_size: int = KERNEL_SIZE,
        stride: int = STRIDE,
    ):
        super().__init__()

        self.prefix = prefix
        self.channels = channels
        self.freq = freq
        self.time_factor = time_factor

        # Fixed kernel parameters for causal_conv2d_k3s2
        assert (
            KERNEL_SIZE == kernel_size
        ), f"conv2d is only implemented for kernel_size={KERNEL_SIZE}"
        self.kernel_size = kernel_size
        assert STRIDE == stride, f"conv2d is only implemented for stride={STRIDE}"
        self.stride = stride
        # WARNING! Cache page sizes have to be uniform in fastconformer.
        # We pad cache across frequency dimension to correspond in size to attention page size: 256 * 512
        self.padded_freq = FASTCONFORMER_CACHE_PAGE_SIZE // self.channels

        # Depthwise conv weight: (kH, kW, C) for custom kernel
        self.conv_weight = nn.Parameter(
            torch.zeros(self.kernel_size, self.kernel_size, self.channels),
            requires_grad=False,
        )
        self.conv_bias = nn.Parameter(
            torch.zeros(self.channels),
            requires_grad=False,
        )

        self.cache_config = cache_config
        self.kv_cache = [torch.tensor([])]
        self.dtype = dtype

        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.conv2d_layer(
            hidden_states,
            self.prefix,
        )

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (T, Freq, Channels) input tensor
        Returns:
            output: (T/2, Freq/2, Channels)
        """
        assert hidden_states.dim() == 3, "forward expects 3D tensor (T, F, C)"
        assert hidden_states.shape[1] == self.freq, f"freq expected {self.freq} but got {hidden_states.shape[1]}"
        assert hidden_states.shape[2] == self.channels, f"channels expected {self.channels} but got {hidden_states.shape[2]}"

        seq_len = hidden_states.shape[0]
        out_seq_len = seq_len // 2
        out_freq = self.freq // 2

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata

        if attn_meta_all is None:
            # Profile run - return zeros with correct output shape
            return torch.zeros(
                (out_seq_len, out_freq, self.channels),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        x = hidden_states.contiguous()

        attn_metadata: FastConformerConvMetadata = attn_meta_all[self.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        # Cache store: (num_blocks, padded_freq, channels)
        # Only [:, :freq, :] is used; padding ensures uniform page sizes
        store = self.kv_cache[fctx.virtual_engine]

        query_start_loc = attn_metadata.query_start_loc

        # True if cache has valid data (decode), False for first call (prefill)
        has_initial_state = torch.ones(
            page_indices.size(0), dtype=torch.bool, device=x.device
        )

        out = torch.empty(
            (out_seq_len, out_freq, self.channels), device=x.device, dtype=x.dtype
        )

        depthwise_strided_conv2d_cached(
            x,
            self.conv_weight,
            self.conv_bias,
            out,
            store,
            query_start_loc,
            page_indices,
            has_initial_state,
            time_factor=self.time_factor,
            output_divisor=self.stride,
            metadata=attn_metadata,
        )
        return out

    @property
    def output_elements_per_token(self) -> int:
        """Number of output time steps produced per input token."""
        return self.time_factor // self.stride

    def get_attn_backend(self) -> AttentionBackend:
        # Use FastConformerConvBackend - kernel handles time_factor and
        # output_divisor internally, same metadata type for all kernels
        return FastConformerConvBackend

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            block_size=1,
            shape=(self.padded_freq, self.channels),
            dtype=self.dtype,
        )


def conv2d_fwd(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.forward_cuda(hidden_states=hidden_states)


def conv2d_fwd_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    # Output shape after stride=2: (seq_len/2, freq/2, channels)
    seq_len, freq, channels = hidden_states.shape
    return torch.zeros(
        (seq_len // 2, freq // 2, channels),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


direct_register_custom_op(
    op_name="conv2d_layer",
    op_func=conv2d_fwd,
    fake_impl=conv2d_fwd_fake,
)


class ConvSubsampling(nn.Module):
    """
    FastConformer convolutional subsampling: 8x time reduction, 80->11 freq.

    Stack: Conv(3x3,s=2) -> ReLU -> [Conv -> Linear -> ReLU] x2 -> Linear
    """

    def __init__(
        self,
        time_factor: int,
        prefix: str,
        cache_config: CacheConfig,
        dtype: torch.dtype,
        n_filt: int = 128,
        channels: int = 256,
        out_dim: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        # pad frequency axis by 1 * time_factor
        # to match dimensions in NeMo implementation
        self.freq_padding = time_factor
        # dimensionality across frequency axis after padding
        cur_freq = n_filt + time_factor

        layers = []
        activation = torch.nn.ReLU(inplace=True)

        # First conv layer
        layers.append(
            Conv2dLayer(
                time_factor,
                prefix=f"{prefix}.conv.0",
                cache_config=cache_config,
                dtype=dtype,
                channels=channels,
                freq=cur_freq,
            )
        )
        layers.append(activation)
        # reduce dimensionality across time and frequency by stride=2
        cur_freq = cur_freq // 2
        time_factor = time_factor // 2

        # Two more conv layers, each followed by pointwise conv
        for i in range(int(math.log2(time_factor))):
            layers.append(
                Conv2dLayer(
                    time_factor,
                    prefix=f"{prefix}.conv.{i+1}",
                    cache_config=cache_config,
                    dtype=dtype,
                    channels=channels,
                    freq=cur_freq,
                )
            )
            cur_freq = cur_freq // 2
            time_factor = time_factor // 2
            layers.append(torch.nn.Linear(channels, channels))
            layers.append(activation)

        self.conv = nn.ModuleList(layers)
        self.out = nn.Linear(channels * cur_freq, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (T_target * Factor, Freq) flattened mel features
        Output: (T_target, out_dim)
        """
        # Pad frequency: 128 + 8 = 136 -> .. -> 17
        x = torch.nn.functional.pad(x, (self.freq_padding, 0))
        # Expand channels: mimic 1->256 conv
        x = x.unsqueeze(2).repeat(1, 1, self.channels).contiguous()

        # Apply conv stack
        for layer in self.conv:
            x = layer(x)

        # Final projection: (T, 11, 256) -> (T, 11*256) -> (T, 512)
        x = self.out(x.transpose(2, 1).flatten(start_dim=1))
        return x


class FastConformerPreprocessor(nn.Module):
    """
    FastConformer preprocessor modules - combines mel spectrogram extraction
    and convolutional subsampling
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # dictionary with all subsampling configuration
        subsampling_conf = vllm_config.model_config.hf_config.subsampling
        hop_length = subsampling_conf.get("hop_length", 160)
        subsampling_factor = subsampling_conf.get("subsampling_factor", 8)
        self.mel_spec = MelSpectrogramLayer(
            time_factor=hop_length * subsampling_factor,
            prefix=f"{prefix}.mel_spec.0",
            cache_config=vllm_config.cache_config,
            dtype=vllm_config.model_config.dtype,
            **subsampling_conf,
        )
        self.pre_encode = ConvSubsampling(
            time_factor=subsampling_factor,
            prefix=prefix,
            cache_config=vllm_config.cache_config,
            dtype=vllm_config.model_config.dtype,
            **subsampling_conf,
        )

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio: (T_target, factor) tensor,
            where factor = 8 (subsampling factor) * 160 (hop length)
        Returns:
            x: (T_target, 512) output
        """
        mel = self.mel_spec(audio.view(-1))  # frames x freq_bins
        emb = self.pre_encode(mel)  # frames/8 x 512
        return emb, mel

    def load_weights(self, nemo: dict[str, torch.Tensor]):
        """Load weights from FastConformer checkpoint."""

        # Initialize mel spectrogram STFT basis and filterbank
        self.mel_spec.init_stft_basis()

        # Depthwise conv layers: (C, 1, kH, kW) -> (kH, kW, C) with flip
        self.pre_encode.conv[0].conv_weight.data.copy_(
            nemo["encoder.pre_encode.conv.0.weight"]
            .squeeze(1)
            .permute(1, 2, 0)
            .flip(1)
            .flip(0)
            .contiguous()
        )
        self.pre_encode.conv[2].conv_weight.data.copy_(
            nemo["encoder.pre_encode.conv.2.weight"]
            .squeeze(1)
            .permute(1, 2, 0)
            .flip(1)
            .flip(0)
        )
        self.pre_encode.conv[5].conv_weight.data.copy_(
            nemo["encoder.pre_encode.conv.5.weight"]
            .squeeze(1)
            .permute(1, 2, 0)
            .flip(1)
            .flip(0)
        )

        # Pointwise conv (1x1): (out, in, 1, 1) -> (out, in)
        self.pre_encode.conv[3].weight.data.copy_(
            nemo["encoder.pre_encode.conv.3.weight"].squeeze(-1).squeeze(-1)
        )
        self.pre_encode.conv[3].bias.data.copy_(nemo["encoder.pre_encode.conv.3.bias"])

        self.pre_encode.conv[6].weight.data.copy_(
            nemo["encoder.pre_encode.conv.6.weight"].squeeze(-1).squeeze(-1)
        )
        self.pre_encode.conv[6].bias.data.copy_(nemo["encoder.pre_encode.conv.6.bias"])

        # Conv biases
        self.pre_encode.conv[0].conv_bias.data.copy_(
            nemo["encoder.pre_encode.conv.0.bias"]
        )
        self.pre_encode.conv[2].conv_bias.data.copy_(
            nemo["encoder.pre_encode.conv.2.bias"]
        )
        self.pre_encode.conv[5].conv_bias.data.copy_(
            nemo["encoder.pre_encode.conv.5.bias"]
        )

        # Output projection
        self.pre_encode.out.weight.data.copy_(nemo["encoder.pre_encode.out.weight"])
        self.pre_encode.out.bias.data.copy_(nemo["encoder.pre_encode.out.bias"])
