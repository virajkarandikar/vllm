# SPDX-License-Identifier: Apache-2.0

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

from .utils import AutoWeightsLoader


@CustomOp.register("toy_conv2d_layer")
class ToyConv2dLayer(CustomOp, AttentionLayerBase):
    """
    Toy 2D convolution layer using causal_conv2d_k3s2 kernel.
    It works for specific usecase: strided convolution is applied to melspec.
    Here we run cached inference with cache over time dimension, while
    both time and frequency dimensions are getting reduced.

    This is a part of bigger model which works on particular time resolution.
    Layer takes a multiplier which describes relation between target and current time resolution.
    
    Input: (T_target x Factor * Freq * Channels)
    Output: (T_target x Factor/2 * Freq/2 * Channels)
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
        self.channels = channels  # num channels
        self.freq = freq
        self.time_factor = time_factor
        
        # Fixed kernel parameters for causal_conv2d_k3s2
        self.kernel_size = 3
        self.stride = 2
        self.padded_freq = 512
        assert self.channels == 256, f"Hardedcoded padded_freq={self.padded_freq} assumes channels==256"
        
        # depthwise separate convolution weight for a custom kernel
        self.conv_weight = nn.Parameter(
            torch.zeros(self.kernel_size,self.kernel_size, self.channels),
            requires_grad=False,
        )
        # TODO: create bias parameter

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
        CUDA forward using the custom convolution kernel.
        
        Args:
            hidden_states: (T_target x Factor * Freq * Channels) input tensor
            
        Returns:
            output: (T_target x Factor/2 * Freq/2 * Channels)
        """
        assert hidden_states.dim() == 2, "forward expects a 2D tensor (T_target x Factor * Freq * Channels)"
        t_target, factor_freq_channels = hidden_states.shape
        assert factor_freq_channels == self.time_factor * self.freq * self.channels, f"hidden state dim {factor_freq_channels} should be a prod of time factor, freq, channels"
        
        hidden_states = hidden_states.view(t_target * self.time_factor, self.freq, self.channels)  # T x Freq x Channels

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata

        if attn_meta_all is None:
            # Profile run - return zeros with correct output shape
            return torch.zeros(
                t_target,
                self.time_factor // 2 * self.freq // 2 * self.channels,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
        
        # Input is already (C, T, F) - matches kernel expectation
        x = hidden_states.contiguous()

        attn_metadata: ToyConv2dMetadata = attn_meta_all[self.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        # store is larger across `freq` dimension, but kernel only uses [idx, :freq] part
        store = self.kv_cache[fctx.virtual_engine]  # (num_blocks, 512, Channels)

        query_start_loc = attn_metadata.query_start_loc
        
        # has_initial_state comes from metadata - True if cache has valid data (decode),
        # False for first call (prefill). This is computed based on num_computed_tokens.
        # has_initial_state = attn_metadata.has_initial_state
        has_initial_state = torch.ones(
            page_indices.size(0), dtype=torch.bool, device=x.device
        )

        # Allocate output tensor before calling kernel
        # Output shape: (cu_time_out, freq//2, channels) where cu_time_out = cu_time_in // 2
        cu_time_in = x.shape[0]
        cu_time_out = cu_time_in // 2
        freq_out = self.freq // 2
        out = torch.empty(
            (cu_time_out, freq_out, self.channels),
            device=x.device,
            dtype=x.dtype
        )

        # Use pre-computed metadata from the builder (no CPU blocking)
        # The backend returned by get_toy_conv2d_backend(time_factor) has a builder
        # that pre-computes metadata with the correct time_factor.
        depthwise_strided_conv2d_cached(
            x,
            self.conv_weight,
            out,
            store,
            query_start_loc,
            page_indices,
            has_initial_state,
            metadata=attn_metadata,
        )  # (time / 2, freq / 2, channels)
        
        # reshape back to (t_target, Factor/2 * Freq/2 * Channels)
        return out.view(t_target, self.time_factor // 2 * self.freq // 2 * self.channels)

    def get_attn_backend(self) -> AttentionBackend:
        # Return a backend configured for this layer's time_factor
        # The backend's builder will pre-compute metadata with the correct time_factor
        return get_toy_conv2d_backend(self.time_factor)

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            # WARNING: when caching is enabled, block size be the same across all layers.
            # for conv we need only 1 though.
            block_size=1,
            # WARNING: this we hardcode a larger cache, so the page has same
            # size as page for fastconformer attn.
            # self.padded_freq is computed assuming certain channels number.
            # during usage we just slice according to self.freq. 
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
    # The fake kernel must return the correct OUTPUT shape, not input shape.
    # Input:  (T_target, time_factor * freq * channels)
    # Output: (T_target, time_factor/2 * freq/2 * channels)
    # Since stride=2 in both time and freq dimensions, output is 1/4 the size.
    t_target = hidden_states.shape[0]
    output_dim = hidden_states.shape[1] // 4
    return torch.zeros(
        t_target, output_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


direct_register_custom_op(
    op_name="toy_conv2d_layer",
    op_func=toy_conv2d_fwd,
    fake_impl=toy_conv2d_fwd_fake,
)


@support_torch_compile
class ToyConv(nn.Module):
    """
    Toy model for testing causal_conv2d_k3s2 kernel.
    
    Uses a single ToyConv2dLayer with kernel=(3,3), stride=(2,2).
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        # TODO: extract the values from config
        self.conv = torch.nn.ModuleList([
            ToyConv2dLayer(
                channels=256,
                freq=80,
                time_factor=4,
                prefix=f"{prefix}.conv.0",
                cache_config=vllm_config.cache_config,
                dtype=vllm_config.model_config.dtype,
            ),
            ToyConv2dLayer(
                channels=256,
                freq=40,
                time_factor=2,
                prefix=f"{prefix}.conv.1",
                cache_config=vllm_config.cache_config,
                dtype=vllm_config.model_config.dtype,
            ),

        ])
        
        # not used, but present for compatability with vLLM generation model
        self.vocab_size = getattr(self.config, "vocab_size", 1)
        self.embed_tokens = nn.Embedding(self.vocab_size, 256)
        self.proj = nn.Linear(256, self.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        conv_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            conv_input: (T_target, Factor * Freq * Channels) tensor
            
        Returns:
            x: (T_target, Factor/2 * Freq/2 * Channels) output
        """
        x = self.conv[0](conv_input)
        x = self.conv[1](x)
        return x, x

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
