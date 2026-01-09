# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Attention backend for varlen chunked kernels (conv2d, STFT, etc.).

Pre-computes batch_ptr and time_chunk_offset_ptr for mapping program IDs
to (sequence_index, chunk_index) pairs, enabling CUDA graph compatibility.
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
    compute_varlen_chunk_metadata,
)

logger = init_logger(__name__)


@dataclass
class VarlenChunkMetadata:
    """
    Metadata for varlen chunked kernels (conv2d, STFT, etc.).
    
    Contains chunk mapping tensors for triton kernels that process
    packed sequences in blocks.
    """
    num_reqs: int  # num_seqs
    query_start_loc: torch.Tensor  # [num_seqs+1]
    slot_mapping: torch.Tensor  # [num_seqs]
    block_table_tensor: torch.Tensor  # [num_seqs, num_blocks]
    kernel_block_size: int  # number of output frames per Triton program

    # Chunk mapping for triton kernels
    batch_ptr: Optional[torch.Tensor] = None  # maps program_id -> sequence index
    time_chunk_offset_ptr: Optional[torch.Tensor] = None  # maps program_id -> chunk index
    query_start_loc_out: Optional[torch.Tensor] = None  # cumulative output positions
    num_programs: Optional[int] = None  # total programs to launch


def _create_builder(
    time_factor: int,
    output_divisor: int,
    kernel_block_size: int,
) -> Type[AttentionMetadataBuilder]:
    """Create a metadata builder class with the specified configuration."""

    class VarlenChunkMetadataBuilder(AttentionMetadataBuilder):
        """Builder for varlen chunk metadata."""
        cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
        _time_factor: ClassVar[int] = time_factor
        _kernel_block_size: ClassVar[int] = kernel_block_size
        _output_divisor: ClassVar[int] = output_divisor

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> VarlenChunkMetadata:
            chunk_meta = compute_varlen_chunk_metadata(
                common_attn_metadata.query_start_loc,
                time_factor=self._time_factor,
                output_divisor=self._output_divisor,
                kernel_block_size=self._kernel_block_size,
            )

            return VarlenChunkMetadata(
                num_reqs=common_attn_metadata.num_reqs,
                query_start_loc=chunk_meta["query_start_loc"],
                slot_mapping=common_attn_metadata.slot_mapping,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                batch_ptr=chunk_meta["batch_ptr"],
                time_chunk_offset_ptr=chunk_meta["time_chunk_offset_ptr"],
                query_start_loc_out=chunk_meta["query_start_loc_out"],
                num_programs=chunk_meta["num_programs"],
                kernel_block_size=self._kernel_block_size,
            )

    # Set a meaningful class name for debugging
    VarlenChunkMetadataBuilder.__name__ = f"VarlenChunkMetadataBuilder_tf{time_factor}_d{output_divisor}"
    VarlenChunkMetadataBuilder.__qualname__ = f"VarlenChunkMetadataBuilder_tf{time_factor}_d{output_divisor}"

    return VarlenChunkMetadataBuilder


@lru_cache(maxsize=32)
def get_varlen_chunk_backend(
    time_factor: int = 1,
    output_divisor: int = 2,
    kernel_block_size: int = 1,
) -> Type[AttentionBackend]:
    """
    Factory function that returns a backend for varlen chunked kernels.
    
    The returned backend pre-computes metadata for kernels that process
    packed sequences in chunks, avoiding CPU blocking during forward.
    
    Output length formula: seqlens_out = (seqlens_in * time_factor) // output_divisor
    
    Examples:
        - Conv2d stride=2: get_varlen_chunk_backend(time_factor=4, output_divisor=2)
        - STFT: get_varlen_chunk_backend(output_divisor=hop_length, block_size=16)
    
    Args:
        time_factor: Multiplier for scaling query_start_loc (default 1)
        output_divisor: Divisor for output length calculation (default 2)
        kernel_block_size: Chunk size for processing (default 64)
        
    Returns:
        A VarlenChunkBackend class configured with the given parameters
    """
    builder_cls = _create_builder(time_factor, output_divisor, kernel_block_size)

    class VarlenChunkBackend(AttentionBackend):
        """Backend for varlen chunked kernels."""

        @staticmethod
        def get_metadata_cls() -> type["AttentionMetadata"]:
            return VarlenChunkMetadata

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
    VarlenChunkBackend.__name__ = f"VarlenChunkBackend_tf{time_factor}_d{output_divisor}"
    VarlenChunkBackend.__qualname__ = f"VarlenChunkBackend_tf{time_factor}_d{output_divisor}"

    return VarlenChunkBackend

