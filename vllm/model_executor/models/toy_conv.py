# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.fastconformer_conv import (
    FastConformerConvBackend,
    FastConformerConvMetadata,
)
from vllm.model_executor.models.fastconformer import ConformerConvModule
from vllm.v1.kv_cache_interface import KVCacheSpec, FastConformerConvSpec
from vllm.model_executor.layers.conv import causal_conv2d_k3s2_fn
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors

from .utils import AutoWeightsLoader


class ToyConv2dLayer(ConformerConvModule):
    """
    Toy 2D convolution layer using causal_conv2d_k3s2 kernel.
    
    This layer uses true 2D convolution:
    - Kernel: (3, 3) over (time, frequency)
    - Stride: (2, 2) in both dimensions
    - Cache: 1 time frame (kernel_t - stride_t = 3 - 2 = 1)
    
    Input: (C, T, F_in) where F_in = 80 (mel frequency bins)
    Output: (C, T_out, F_out) where:
        - T_out = (T + 1) // 2  (with cache providing time position -1)
        - F_out = (F_in - 3) // 2 + 1 = 39 for F_in=80
    
    The weights are stored in nn.Conv2d for easy comparison with PyTorch:
        # PyTorch reference (non-causal, for testing):
        # Pad input with 1 frame on left (time) to simulate cache=0
        # x_padded = F.pad(x, (0, 0, 1, 0))  # [B, C, T+1, F]
        # y = self.conv2d(x_padded)  # [B, C, T_out, F_out]
    """
    
    def __init__(
        self,
        d_model: int,  # number of channels (C)
        freq_in: int,  # input frequency dimension (typically 80)
        prefix: str,
        cache_config: CacheConfig,
        dtype: torch.dtype,
    ):
        # Do not call super().__init__ to avoid creating unused submodules
        nn.Module.__init__(self)
        
        self.prefix = prefix
        self.d_model = d_model  # num channels
        self.freq_in = freq_in
        
        # Fixed kernel parameters for causal_conv2d_k3s2
        self.kernel_size = (3, 3)  # (time, frequency)
        self.stride = (2, 2)
        self.cache_t = 1  # time frames to cache = kernel_t - stride_t
        
        # Output frequency dimension
        self.freq_out = (freq_in - 3) // 2 + 1
        
        # Depthwise Conv2d: groups=d_model for depthwise
        # Shape: (d_model, 1, 3, 3) for depthwise
        # We use groups=d_model so each channel has its own 3x3 kernel
        self.conv2d = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # No padding - handled by cache in time dimension
            groups=d_model,  # Depthwise
            bias=True,
            dtype=dtype,
        )
        
        # Cache shape: (cache_t * block_size, d_model, freq_in)
        # Stores 1 time frame per channel with full frequency
        self.left_shape = self.cache_t * cache_config.block_size

        self.cache_config = cache_config
        self.kv_cache = [torch.tensor([])]
        self.dtype = dtype

        try:
            config = get_current_vllm_config()
            if config is not None:
                compilation = config.compilation_config
                if prefix not in compilation.static_forward_context:
                    compilation.static_forward_context[prefix] = self
        except Exception:
            pass

    def get_kernel_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract weights in the format expected by causal_conv2d_k3s2_fn.
        
        Returns:
            weight: (C, 3, 3) tensor - depthwise 2D kernel
            bias: (C,) tensor
        """
        # conv2d.weight shape for depthwise: (C, 1, 3, 3)
        # We need: (C, 3, 3)
        weight = self.conv2d.weight.squeeze(1)  # (C, 3, 3)
        bias = self.conv2d.bias  # (C,)
        return weight, bias

    def pytorch_reference(
        self, 
        x: torch.Tensor, 
        cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        PyTorch reference implementation for testing.
        
        Args:
            x: (C, T, F) input tensor
            cache: (C, 1, F) cached time frame or None (treated as zeros)
            
        Returns:
            output: (C, T_out, F_out) tensor
        """
        C, T, F = x.shape
        assert F == self.freq_in
        
        # Reshape to (1, C, T, F) for conv2d
        x_4d = x.unsqueeze(0)  # (1, C, T, F)
        
        # Prepend cache (or zeros) for causal padding in time
        if cache is None:
            cache = torch.zeros(1, C, 1, F, dtype=x.dtype, device=x.device)
        else:
            cache = cache.unsqueeze(0)  # (1, C, 1, F)
        
        # Concatenate cache + input: (1, C, T+1, F)
        x_padded = torch.cat([cache, x_4d], dim=2)
        
        # Apply conv2d with stride=(2,2), kernel=(3,3)
        y = self.conv2d(x_padded)  # (1, C, T_out, F_out)
        
        # Remove batch dimension
        return y.squeeze(0)  # (C, T_out, F_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using causal_conv2d_k3s2 kernel.
        
        Args:
            hidden_states: (C, T, F) input tensor
            
        Returns:
            output: (C, T_out, F_out) tensor after 2D strided convolution
        """
        assert hidden_states.dim() == 3, "forward expects a 3D tensor (C, T, F)"
        C, T, F = hidden_states.shape
        #assert C == self.d_model
        assert F == self.freq_in

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata

        if attn_meta_all is None:
            # Profile run - return zeros with correct output shape
            T_out = (T + 1) // 2
            return torch.zeros(C, T_out, self.freq_out, 
                             dtype=hidden_states.dtype, 
                             device=hidden_states.device)

        # Get weights from Conv2d in kernel format
        conv_weights, conv_bias = self.get_kernel_weights()
        
        # Input is already (C, T, F) - matches kernel expectation
        x = hidden_states.contiguous()

        attn_metadata: FastConformerConvMetadata = attn_meta_all[self.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        # Get conv state cache: (num_cache_lines, C, 1, F)
        store = self.kv_cache[fctx.virtual_engine]
        # The cache is stored as (num_lines, left_shape, C, F)? 
        # We need (num_lines, C, 1, F) for the kernel
        # Assuming cache layout matches what we need
        conv_state = store[:, :self.cache_t, :, :].transpose(1, 2).contiguous()

        query_start_loc = attn_metadata.query_start_loc
        has_initial_state = torch.ones(
            page_indices.size(0), dtype=torch.bool, device=x.device
        )

        # Call the 2D causal conv kernel
        y = causal_conv2d_k3s2_fn(
            x=x,                                  # (C, cu_time, F)
            weight=conv_weights,                  # (C, 3, 3)
            bias=conv_bias,                       # (C,)
            conv_state=conv_state,                # (num_cache_lines, C, 1, F)
            query_start_loc=query_start_loc,      # (batch + 1,)
            cache_indices=page_indices,           # (batch,)
            has_initial_state=has_initial_state,  # (batch,)
            activation=None,
            metadata=attn_metadata,
        )
        
        return y  # (C, T_out, F_out)

    def get_attn_backend(self) -> AttentionBackend:
        return FastConformerConvBackend

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            block_size=self.cache_config.block_size,
            shape=(self.left_shape, self.d_model, self.freq_in),
            dtype=self.dtype,
        )


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
        self.d_model = self.config.d_model
        
        # Frequency input dimension (mel bins)
        #self.freq_in = getattr(self.config, "freq_in", 80)
        self.freq_in = 82

        self.conv = nn.ModuleList([
            ToyConv2dLayer(
                d_model=self.d_model,
                freq_in=self.freq_in,
                prefix=f"{prefix}.conv.0",
                cache_config=vllm_config.cache_config,
                dtype=vllm_config.model_config.dtype,
            )
        ])
        
        self.vocab_size = getattr(self.config, "vocab_size", 1)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        self.proj = nn.Linear(self.d_model, self.vocab_size)

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
            conv_input: (T, F) tensor
            
        Returns:
            x: (C, T_out, F_out) output after 2D strided convolution
        """
        conv_input = conv_input.unsqueeze(0)  # 1 x T x F
        x = self.conv[0](conv_input)  # C x time x freq
        x = x.transpose(0, 1)  # time x C x freq
        x = x.flatten(1)  # time x C*freq
        return x, x

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
