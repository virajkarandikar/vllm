#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn.functional as F

from vllm.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
reshape_and_cache_flash = triton_reshape_and_cache_flash

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.platforms import current_platform

logger = init_logger(__name__)



class FastConformerRPEBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_quant_query_input: bool = False

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        return  # supports any head size

    @staticmethod
    def get_name() -> str:
        return "FASTCONF_RPE"

    @staticmethod
    def get_impl_cls() -> type["FastConformerRPEImpl"]:
        return FastConformerRPEImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FastConformerRPEMetadata

    @staticmethod
    def get_builder_cls() -> type["FastConformerRPEMetadataBuilder"]:
        return FastConformerRPEMetadataBuilder

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
class FastConformerRPEMetadata:
    num_actual_tokens: int
    num_reqs: int
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    doc_ids: torch.Tensor
    decode_offset: torch.Tensor
    causal: bool = True


class FastConformerRPEMetadataBuilder(AttentionMetadataBuilder[FastConformerRPEMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FastConformerRPEMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        decode_offset = common_attn_metadata.decode_offset
        doc_ids = common_attn_metadata.doc_ids
        return FastConformerRPEMetadata(
            num_actual_tokens=num_actual_tokens,
            num_reqs=num_reqs,
            query_start_loc=query_start_loc,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            doc_ids=doc_ids,
            decode_offset=decode_offset,
            causal=common_attn_metadata.causal,
        )


class FastConformerRPEImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        window: Optional[int] = None,
        pos_bias_u: Optional[torch.Tensor] = None,
        pos_bias_v: Optional[torch.Tensor] = None,
        linear_pos: Optional[torch.nn.Linear] = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi not supported for FastConformer RPE.")
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("Quantized KV cache not supported for FastConformer RPE.")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV-sharing not supported in FastConformer RPE.")

        self.window = window
        self.h = num_heads
        self.dh = head_size
        self.pos_bias_u = pos_bias_u
        self.pos_bias_v = pos_bias_v
        self.linear_pos = linear_pos

        self._zero_scalar = {
            torch.float16: torch.tensor(0, dtype=torch.float16, device=torch.device("cuda")),
            torch.bfloat16: torch.tensor(0, dtype=torch.bfloat16, device=torch.device("cuda")),
            torch.float32: torch.tensor(0, dtype=torch.float32,  device=torch.device("cuda")),
        }

        self._zero_long  = torch.tensor(0, dtype=torch.long, device=torch.device("cuda"))
        self._max_tokens = 1310720
        self._max_kcap = 2048
        self._arange     = torch.arange(self._max_tokens, device=torch.device("cuda"), dtype=torch.long)
        self._slotbuf    = torch.arange(self._max_kcap, device=torch.device("cuda"), dtype=torch.long)

        self._rel_cache: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

        assert self.num_heads == self.num_kv_heads, (
            "FastConformer RPE assumes num_heads == num_kv_heads."
        )

    def _get_rel_proj(self, device: torch.device, dtype: torch.dtype, W: int) -> torch.Tensor:
        key = (device, dtype, W)
        cached = self._rel_cache.get(key)
        if cached is not None:
            return cached

        H, Dh = self.h, self.dh
        D = H * Dh

        deltas = torch.arange(-W, W + 1, device=device)[:, None].to(torch.float32)
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / D))
        sin = torch.sin(deltas * div)
        cos = torch.cos(deltas * div)
        rel = torch.zeros((2 * W + 1, D), device=device, dtype=torch.float32)
        rel[:, 0::2] = sin
        rel[:, 1::2] = cos

        rel = self.linear_pos(rel.to(dtype))              # [2W+1, D]
        rel = rel.view(2 * W + 1, H, Dh).permute(1, 2, 0).contiguous()  # [H, Dh, 2W+1]
        self._rel_cache[key] = rel
        return rel

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FastConformerRPEMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [num_tokens, H, Dh]
            key:   [num_tokens, H, Dh]
            value: [num_tokens, H, Dh]
            kv_cache: [2, num_blocks, block_size, H, Dh]
        Returns:
            [num_tokens, H, Dh] if output provided, else flattened
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Output scaling not supported for FastConformer RPE.")

        if attn_metadata is None:
            return output

        q_u = query + self.pos_bias_u.unsqueeze(0)
        q_v = query + self.pos_bias_v.unsqueeze(0)

        block_size = int(attn_metadata.block_size)
        N_live = int(attn_metadata.num_actual_tokens)

        W = 71
        assert W < block_size, f"window ({W}) must be < cache block_size ({block_size})"
        K_cap = W + 1
        H = self.num_heads
        Dh = self.head_size

        key_cache, value_cache = kv_cache.unbind(0)  # [num_blocks, block, H, Dh]
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        req_ids = attn_metadata.doc_ids[:N_live].to(torch.long)
        q_start = attn_metadata.query_start_loc.index_select(0, req_ids)
        dec_off = attn_metadata.decode_offset.index_select(0, req_ids)

        device = q_u.device
        idx  = self._arange[:N_live]
        local_q = idx - q_start
        logical_q = local_q + dec_off

        start = torch.clamp(logical_q - W, min=0)
        L = logical_q - start + 1
        pad = K_cap - L

        start_block = torch.div(start, block_size, rounding_mode="floor")
        end_block = torch.div(logical_q, block_size, rounding_mode="floor")
        start_off = start - start_block * block_size
        end_off = logical_q - end_block * block_size

        bt = attn_metadata.block_table
        phys_start = bt.index_select(0, req_ids).gather(1, start_block.unsqueeze(1)).squeeze(1)
        phys_end = bt.index_select(0, req_ids).gather(1, end_block.unsqueeze(1)).squeeze(1)

        two_block = (start_block != end_block)

        slot = self._slotbuf[:K_cap].unsqueeze(0).expand(N_live, K_cap)
        valid = slot >= pad.unsqueeze(1)
        relpos = slot - pad.unsqueeze(1)
        thresh = (block_size - start_off).unsqueeze(1)

        kv_block_start = phys_start.unsqueeze(1).expand_as(slot)
        kv_off_start = start_off.unsqueeze(1) + relpos
        kv_block_end = phys_end.unsqueeze(1).expand_as(slot)
        kv_off_end = relpos - thresh

        use_end = two_block.unsqueeze(1) & (relpos >= thresh)

        has_start = (phys_start >= 0)
        has_end   = (phys_end   >= 0)
        valid_block = torch.where(use_end, has_end.unsqueeze(1), has_start.unsqueeze(1))
        valid = valid & valid_block

        kv_block = torch.where(use_end, kv_block_end, kv_block_start)
        kv_off   = torch.where(use_end, kv_off_end,   kv_off_start)
        kv_off   = kv_off.clamp_(min=0, max=block_size - 1)

        safe_block = kv_block_start.clamp_min(0)
        safe_off   = self._zero_long

        kv_block = torch.where(valid, kv_block, safe_block)
        kv_off   = torch.where(valid,   kv_off,  safe_off)

        ks = key_cache[kv_block, kv_off]    # [N, K_cap, H, Dh]
        vs = value_cache[kv_block, kv_off]  # [N, K_cap, H, Dh]

        mask4 = valid.unsqueeze(2).unsqueeze(3)
        zero_k = self._zero_scalar[ks.dtype]
        zero_v = self._zero_scalar[vs.dtype]

        ks = torch.where(mask4, ks, zero_k)
        vs = torch.where(mask4, vs, zero_v)

        rel = self._get_rel_proj(device, q_u.dtype, W)  # [H, Dh, 2W+1]
        band = torch.einsum("n h d, h d m -> n h m", q_v[:N_live], rel) * (Dh ** -0.5)
        band_idx = (W + (L.unsqueeze(1) - 1) - relpos).clamp(0, 2 * W)
        bias_g = band.gather(2, band_idx.unsqueeze(1).expand(-1, H, -1))
        bias_g = bias_g.masked_fill(~valid.unsqueeze(1), float("-inf"))

        attn_bias_sdpa = bias_g.reshape(N_live * H, 1, K_cap).contiguous()
        q_sdpa = q_u[:N_live].reshape(N_live * H, 1, Dh)
        k_sdpa = ks.permute(0, 2, 1, 3).reshape(N_live * H, K_cap, Dh)
        v_sdpa = vs.permute(0, 2, 1, 3).reshape(N_live * H, K_cap, Dh)

        y = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_bias_sdpa, dropout_p=0.0, is_causal=False
        )  # [N*H, 1, Dh]

        y = y.view(N_live, H, Dh)
        output.zero_()
        output[:N_live].copy_(y)
        return output
