# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Attention backend for ToyConv2d - uses causal_conv2d_k3s2 kernel.
Separate from FastConformerConvBackend to keep metadata lightweight.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Optional, Type

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv2d_metadata,
)

logger = init_logger(__name__)


@dataclass
class ToyConv2dMetadata:
    """Metadata for causal_conv2d_k3s2 used in ToyConv."""
    num_reqs: int  # num_seqs
    query_start_loc: torch.Tensor  # [num_seqs+1]
    slot_mapping: torch.Tensor  # [num_seqs]
    block_table_tensor: torch.Tensor  # [num_seqs, num_blocks]

    # these attributes are for triton implementation of causal_conv2d_k3s2
    batch_ptr: Optional[torch.Tensor] = None
    time_chunk_offset_ptr: Optional[torch.Tensor] = None
    query_start_loc_out: Optional[torch.Tensor] = None
    num_programs: Optional[int] = None


def _create_toy_conv2d_builder(time_factor: int) -> Type[AttentionMetadataBuilder]:
    """Create a metadata builder class with the specified time_factor."""

    class ToyConv2dMetadataBuilder(AttentionMetadataBuilder):
        """Builder for ToyConv2d metadata with time_factor={time_factor}."""
        cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
        _time_factor: ClassVar[int] = time_factor

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> ToyConv2dMetadata:
            # Compute conv2d metadata with the configured time_factor
            conv2d_meta = compute_causal_conv2d_metadata(
                common_attn_metadata.query_start_loc,
                time_factor=self._time_factor,
            )

            return ToyConv2dMetadata(
                num_reqs=common_attn_metadata.num_reqs,
                query_start_loc=common_attn_metadata.query_start_loc,
                slot_mapping=common_attn_metadata.slot_mapping,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                batch_ptr=conv2d_meta["batch_ptr"],
                time_chunk_offset_ptr=conv2d_meta["time_chunk_offset_ptr"],
                query_start_loc_out=conv2d_meta["query_start_loc_out"],
                num_programs=conv2d_meta["num_programs"],
            )

    # Set a meaningful class name for debugging
    ToyConv2dMetadataBuilder.__name__ = f"ToyConv2dMetadataBuilder_tf{time_factor}"
    ToyConv2dMetadataBuilder.__qualname__ = f"ToyConv2dMetadataBuilder_tf{time_factor}"

    return ToyConv2dMetadataBuilder


@lru_cache(maxsize=16)
def get_toy_conv2d_backend(time_factor: int) -> Type[AttentionBackend]:
    """
    Factory function that returns a ToyConv2dBackend class configured for
    the specified time_factor.
    
    The returned backend's builder will pre-compute metadata with the correct
    time_factor, avoiding CPU blocking during forward.
    
    Args:
        time_factor: The time scaling factor for this conv layer
        
    Returns:
        A ToyConv2dBackend class with builder configured for the given time_factor
    """
    builder_cls = _create_toy_conv2d_builder(time_factor)

    class ToyConv2dBackend(AttentionBackend):
        """Backend for ToyConv2d using causal_conv2d_k3s2 kernel (time_factor={time_factor})."""

        @staticmethod
        def get_metadata_cls() -> type["AttentionMetadata"]:
            return ToyConv2dMetadata

        @classmethod
        def get_supported_head_sizes(cls) -> list[int]:
            return [32, 64, 96, 128, 160, 192, 224, 256]

        @staticmethod
        def get_builder_cls() -> type[AttentionMetadataBuilder]:
            return builder_cls

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            return (2, num_blocks, block_size, num_kv_heads, head_size)

    # Set meaningful class name for debugging
    ToyConv2dBackend.__name__ = f"ToyConv2dBackend_tf{time_factor}"
    ToyConv2dBackend.__qualname__ = f"ToyConv2dBackend_tf{time_factor}"

    return ToyConv2dBackend

