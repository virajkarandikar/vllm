# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


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
    num_reqs: int  # num_seqs
    query_start_loc: torch.Tensor  # [num_seqs+1]
    slot_mapping: torch.Tensor  # [num_seqs]
    block_table_tensor: torch.Tensor  # [num_seqs, num_blocks]

    # These attributes are for triton implementation of causal_conv1d.
    # nums_dict is structured as expected by causal_conv1d_fn.
    nums_dict: Optional[dict] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


class FastConformerConvMetadataBuilder(AttentionMetadataBuilder):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    # Block size used in causal_conv1d triton kernel
    BLOCK_M: ClassVar[int] = 8

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config

        # Estimate max number of programs needed.
        # Each sequence can have at most ceil(max_model_len / BLOCK_M) chunks.
        max_model_len = vllm_config.model_config.max_model_len
        max_chunks_per_seq = (max_model_len + self.BLOCK_M - 1) // self.BLOCK_M
        self.max_num_programs = max(1024, self.max_num_seqs * max_chunks_per_seq) * 2

        # Pre-allocate CPU (pinned) scratch buffers to avoid per-call allocations.
        self.seqlens_cpu = torch.empty(
            (self.max_num_seqs,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.nums_cpu = torch.empty(
            (self.max_num_seqs,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._tmp_cpu = torch.empty(
            (self.max_num_seqs,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.mlist_cpu = torch.empty(
            (self.max_num_programs,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.offset_cpu = torch.empty(
            (self.max_num_programs,), dtype=torch.int32, device="cpu", pin_memory=True
        )

        # Pre-allocate GPU buffers for causal_conv1d metadata
        self.batch_ptr = torch.full(
            (self.max_num_programs,),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        self.token_chunk_offset_ptr = torch.full(
            (self.max_num_programs,),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )

        # Pre-allocate the nums_dict structure that causal_conv1d_fn expects.
        # This dict is keyed by BLOCK_M and contains pre-allocated tensors.
        self.nums_dict: dict = {
            self.BLOCK_M: {
                "nums": self.nums_cpu[:0],
                "tot": 0,
                "mlist": self.mlist_cpu[:0],
                "mlist_len": 0,
                "offsetlist": self.offset_cpu[:0],
                "batch_ptr": self.batch_ptr,
                "token_chunk_offset_ptr": self.token_chunk_offset_ptr,
            }
        }

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FastConformerConvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        # Compute sequence lengths on CPU using preallocated buffers.
        seqlens = self.seqlens_cpu[:num_reqs]
        torch.sub(
            query_start_loc_cpu[1 : num_reqs + 1],
            query_start_loc_cpu[:num_reqs],
            out=seqlens,
        )

        # Ceil divide by BLOCK_M without creating new tensors.
        tmp = self._tmp_cpu[:num_reqs]
        torch.add(seqlens, self.BLOCK_M - 1, out=tmp)
        nums = self.nums_cpu[:num_reqs]
        torch.div(tmp, self.BLOCK_M, rounding_mode="floor", out=nums)

        total_programs = int(nums.sum().item())

        if total_programs > 0:
            mlist_len = total_programs

            # Resize buffers if needed (should be rare after warmup)
            if mlist_len > self.max_num_programs:
                new_size = mlist_len * 2
                self.mlist_cpu = torch.empty(
                    (new_size,), dtype=torch.int32, device="cpu", pin_memory=True
                )
                self.offset_cpu = torch.empty(
                    (new_size,), dtype=torch.int32, device="cpu", pin_memory=True
                )
                self.batch_ptr = torch.full(
                    (new_size,),
                    PAD_SLOT_ID,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.token_chunk_offset_ptr = torch.full(
                    (new_size,),
                    PAD_SLOT_ID,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.max_num_programs = new_size
                # Update refs in nums_dict
                self.nums_dict[self.BLOCK_M]["batch_ptr"] = self.batch_ptr
                self.nums_dict[self.BLOCK_M]["token_chunk_offset_ptr"] = (
                    self.token_chunk_offset_ptr
                )

            # Build batch mapping and offsets in-place on CPU.
            pos = 0
            for seq_idx in range(num_reqs):
                count = int(nums[seq_idx].item())
                if count == 0:
                    continue
                end = pos + count
                self.mlist_cpu[pos:end].fill_(seq_idx)
                torch.arange(count, out=self.offset_cpu[pos:end])
                pos = end

            # Copy to pre-allocated GPU buffers (non-blocking)
            self.batch_ptr[:mlist_len].copy_(
                self.mlist_cpu[:mlist_len], non_blocking=True
            )
            self.token_chunk_offset_ptr[:mlist_len].copy_(
                self.offset_cpu[:mlist_len], non_blocking=True
            )

            # Update the nums_dict in-place with computed values
            self.nums_dict[self.BLOCK_M]["nums"] = nums
            self.nums_dict[self.BLOCK_M]["tot"] = mlist_len
            self.nums_dict[self.BLOCK_M]["mlist"] = self.mlist_cpu[:mlist_len]
            self.nums_dict[self.BLOCK_M]["mlist_len"] = mlist_len
            self.nums_dict[self.BLOCK_M]["offsetlist"] = self.offset_cpu[:mlist_len]
        else:
            # Empty batch case
            self.nums_dict[self.BLOCK_M]["nums"] = nums
            self.nums_dict[self.BLOCK_M]["tot"] = 0
            self.nums_dict[self.BLOCK_M]["mlist"] = self.mlist_cpu[:0]
            self.nums_dict[self.BLOCK_M]["mlist_len"] = 0
            self.nums_dict[self.BLOCK_M]["offsetlist"] = self.offset_cpu[:0]

        return FastConformerConvMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            query_start_loc=query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            nums_dict=self.nums_dict,
            batch_ptr=self.batch_ptr,
            token_chunk_offset_ptr=self.token_chunk_offset_ptr,
        )
