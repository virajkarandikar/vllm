# SPDX-License-Identifier: Apache-2.0
"""
Cached 2D Convolution Model for FastConformer-style mel-spectrogram subsampling.

Implements strided depthwise conv layers with KV-cache-like state management
for streaming inference. Downsamples (T*8, 80) mel features to (T, 512).
"""

from typing import Optional, Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.toy_conv2d import (
    get_toy_conv2d_backend,
    ToyConv2dMetadata,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.custom_op import CustomOp
from vllm.v1.kv_cache_interface import KVCacheSpec, FastConformerConvSpec
from vllm.model_executor.layers.conv import depthwise_strided_conv2d_cached
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.utils import direct_register_custom_op
from vllm.compilation.decorators import support_torch_compile


@CustomOp.register("toy_conv2d_layer")
class ToyConv2dLayer(CustomOp, AttentionLayerBase):
    """
    Cached depthwise strided 2D convolution layer with kernel=3, stride=2.
    
    Processes (T, Freq, Channels) -> (T/2, Freq/2, Channels).
    Maintains cache for causal streaming over time dimension.
    """
    
    def __init__(
        self,
        channels: int,
        freq: int,
        time_factor: int,
        prefix: str,
        cache_config: CacheConfig,
        dtype: torch.dtype,
    ):
        super().__init__()
        
        self.prefix = prefix
        self.channels = channels
        self.freq = freq
        self.time_factor = time_factor
        
        # Fixed kernel parameters for causal_conv2d_k3s2
        self.kernel_size = 3
        self.stride = 2
        # Padded to 512 so cache pages have same size as fastconformer attention
        self.padded_freq = 512
        assert self.channels == 256, \
            f"padded_freq={self.padded_freq} assumes channels==256"
        
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
        return torch.ops.vllm.toy_conv2d_layer(
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
        assert hidden_states.shape[1] == self.freq
        assert hidden_states.shape[2] == self.channels
        
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
                device=hidden_states.device
            )
        
        x = hidden_states.contiguous()

        attn_metadata: ToyConv2dMetadata = attn_meta_all[self.prefix]
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
            (out_seq_len, out_freq, self.channels),
            device=x.device,
            dtype=x.dtype
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
            metadata=attn_metadata,
        )
        return out

    def get_attn_backend(self) -> AttentionBackend:
        return get_toy_conv2d_backend(self.time_factor)

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            block_size=1,
            shape=(self.padded_freq, self.channels),
            dtype=self.dtype,
        )


def toy_conv2d_fwd(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.forward_cuda(hidden_states=hidden_states)


def toy_conv2d_fwd_fake(
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
    op_name="toy_conv2d_layer",
    op_func=toy_conv2d_fwd,
    fake_impl=toy_conv2d_fwd_fake,
)


class ConvSubsampling(nn.Module):
    """
    FastConformer convolutional subsampling: 8x time reduction, 80->11 freq.
    
    Stack: Conv(3x3,s=2) -> ReLU -> [Conv -> Linear -> ReLU] x2 -> Linear
    """
    
    def __init__(self, hf_config, cache_config, dtype, prefix: str = ""):
        super().__init__()
        self.config = hf_config

        self.freq_no_pad = 80   # Input mel bins
        self.orig_freq = 88     # After padding (divisible by 8)
        self.freq_out = 11      # Output freq after 3x stride-2
        self.total_time_factor = 8
        self.channels = 256
        self.padding = self.orig_freq - self.freq_no_pad
        out_dim = 512
        
        layers = []
        activation = torch.nn.ReLU(inplace=True)
        time_factor = self.total_time_factor
        freq = self.orig_freq

        # First conv layer
        layers.append(
            ToyConv2dLayer(
                channels=self.channels,
                freq=freq,
                time_factor=time_factor,
                prefix=f"{prefix}.conv.0",
                cache_config=cache_config,
                dtype=dtype,
            )
        )
        freq = freq // 2
        time_factor = time_factor // 2
        layers.append(activation)

        # Two more conv layers, each followed by pointwise conv
        for i in range(2):
            layers.append(
                ToyConv2dLayer(
                    channels=self.channels,
                    freq=freq,
                    time_factor=time_factor,
                    prefix=f"{prefix}.conv.{i+1}",
                    cache_config=cache_config,
                    dtype=dtype,
                )
            )
            freq = freq // 2
            time_factor = time_factor // 2
            layers.append(torch.nn.Linear(self.channels, self.channels))
            layers.append(activation)
        
        self.conv = nn.ModuleList(layers)
        self.out = nn.Linear(self.channels * self.freq_out, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (T_target, Factor * Freq) flattened mel features
        Output: (T_target, out_dim)
        """
        # Reshape to (T*factor, freq_no_pad)
        x = x.view(-1, self.freq_no_pad)
        # Pad frequency: 80 + 8 = 88 -> 44 -> 22 -> 11
        x = torch.nn.functional.pad(x, (self.padding, 0))
        # Expand channels: mimic 1->256 conv
        x = x.unsqueeze(2).repeat(1, 1, self.channels).contiguous()

        # Apply conv stack
        for layer in self.conv:
            x = layer(x)

        # Final projection: (T, 11, 256) -> (T, 11*256) -> (T, 512)
        x = self.out(x.transpose(2, 1).flatten(start_dim=1))
        return x


@support_torch_compile
class ToyConv(nn.Module):
    """Toy model for testing cached strided conv2d kernel."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config

        self.pre_encode = ConvSubsampling(
            hf_config=hf_config,
            cache_config=vllm_config.cache_config,
            dtype=vllm_config.model_config.dtype,
            prefix=prefix,
        )
        
        # Compatibility with vLLM generation interface
        self.vocab_size = 1
        self.embed_tokens = nn.Embedding(self.vocab_size, 256)
        self.proj = nn.Linear(512, self.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        conv_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            conv_input: (T_target, Factor * Freq) tensor
        Returns:
            x: (T_target, OutDim) output
        """
        x = self.pre_encode(conv_input)
        return x, x

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from FastConformer checkpoint."""
        nemo = {name: tensor for name, tensor in weights}
        
        # Depthwise conv layers: (C, 1, kH, kW) -> (kH, kW, C) with flip
        self.pre_encode.conv[0].conv_weight.data.copy_(
            nemo["encoder.pre_encode.conv.0.weight"]
            .squeeze(1).permute(1, 2, 0).flip(1).flip(0).contiguous())
        self.pre_encode.conv[2].conv_weight.data.copy_(
            nemo["encoder.pre_encode.conv.2.weight"]
            .squeeze(1).permute(1, 2, 0).flip(1).flip(0))
        self.pre_encode.conv[5].conv_weight.data.copy_(
            nemo["encoder.pre_encode.conv.5.weight"]
            .squeeze(1).permute(1, 2, 0).flip(1).flip(0))
        
        # Pointwise conv (1x1): (out, in, 1, 1) -> (out, in)
        self.pre_encode.conv[3].weight.data.copy_(
            nemo["encoder.pre_encode.conv.3.weight"].squeeze(-1).squeeze(-1))
        self.pre_encode.conv[3].bias.data.copy_(
            nemo["encoder.pre_encode.conv.3.bias"])
        
        self.pre_encode.conv[6].weight.data.copy_(
            nemo["encoder.pre_encode.conv.6.weight"].squeeze(-1).squeeze(-1))
        self.pre_encode.conv[6].bias.data.copy_(
            nemo["encoder.pre_encode.conv.6.bias"])

        # Conv biases
        self.pre_encode.conv[0].conv_bias.data.copy_(
            nemo["encoder.pre_encode.conv.0.bias"])
        self.pre_encode.conv[2].conv_bias.data.copy_(
            nemo["encoder.pre_encode.conv.2.bias"])
        self.pre_encode.conv[5].conv_bias.data.copy_(
            nemo["encoder.pre_encode.conv.5.bias"])
        
        # Output projection
        self.pre_encode.out.weight.data.copy_(nemo["encoder.pre_encode.out.weight"])
        self.pre_encode.out.bias.data.copy_(nemo["encoder.pre_encode.out.bias"])
