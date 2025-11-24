# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
)
from typing import ClassVar, Optional

logger = init_logger(__name__)


class FastConformerConvBackend(AttentionBackend):
    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FastConformerConvMetadata

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_builder_cls() -> type["FastConformerConvMetadataBuilder"]:
        return FastConformerConvMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)




@dataclass
class FastConformerConvMetadata:
    num_reqs: int # num_seqs
    query_start_loc: torch.Tensor # [num_seqs+1]
    slot_mapping: torch.Tensor # [num_seqs]
    block_table_tensor: torch.Tensor # [num_seqs, num_blocks]
    
    # these attributes are for triton implementation of causal_conv1d
    nums_dict: Optional[dict] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


class FastConformerConvMetadataBuilder(AttentionMetadataBuilder):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FastConformerConvMetadata:
        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            common_attn_metadata.query_start_loc,
        )
        return FastConformerConvMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
