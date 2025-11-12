#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn.functional as F
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

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


@triton.jit
def _fc_single_q_attn_kernel(
    Q,          # [NH, Dh] contiguous
    K,          # [NH, Dh, Kcap] contiguous
    V,          # [NH, Kcap, Dh] contiguous
    BIAS,       # [NH, Kcap] contiguous (includes -inf for invalid)
    OUT,        # [NH, Dh] contiguous
    scale,      # float
    Dh,         # int (runtime)
    Kcap,       # int (runtime)
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # 0..NH-1

    q_row = Q + pid * Dh
    k_base = K + pid * (Dh * Kcap)
    v_base = V + pid * (Kcap * Dh)
    b_row = BIAS + pid * Kcap
    o_row = OUT + pid * Dh

    d_offsets = tl.arange(0, BLOCK_D)

    q = tl.load(q_row + d_offsets, mask=d_offsets < Dh, other=0.0).to(tl.float32)

    m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for ko in range(0, Kcap, BLOCK_K):
        k_offsets = ko + tl.arange(0, BLOCK_K)

        k_ptrs = k_base + d_offsets[:, None] * Kcap + k_offsets[None, :]
        k_mask = (d_offsets[:, None] < Dh) & (k_offsets[None, :] < Kcap)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        s = tl.sum(k_block * q[:, None], axis=0) * scale

        b = tl.load(b_row + k_offsets, mask=k_offsets < Kcap, other=-float("inf")).to(tl.float32)
        s = s + b

        m_ij = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_ij)
        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p, axis=0)

        v_ptrs = v_base + k_offsets[:, None] * Dh + d_offsets[None, :]
        v_mask = (k_offsets[:, None] < Kcap) & (d_offsets[None, :] < Dh)
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        pv = v_block * p[:, None]
        sum_pv = tl.sum(pv, axis=0)
        acc = acc * alpha + sum_pv

        m_i = m_ij
        l_i = l_ij

    out = acc / l_i
    tl.store(o_row + d_offsets, out.to(tl.float32), mask=d_offsets < Dh)


def _launch_fc_single_q_attn_triton(
    q_flat: torch.Tensor,        # [NH, Dh], contiguous, CUDA
    k_flat: torch.Tensor,        # [NH, Dh, Kcap], contiguous, CUDA
    v_flat: torch.Tensor,        # [NH, Kcap, Dh], contiguous, CUDA
    bias_flat: torch.Tensor,     # [NH, Kcap], contiguous, CUDA
    scale: float,
) -> torch.Tensor:
    assert q_flat.is_cuda and k_flat.is_cuda and v_flat.is_cuda and bias_flat.is_cuda
    NH, Dh = q_flat.shape
    Kcap = k_flat.shape[2]
    BLOCK_D = 128 if Dh > 64 else 64
    BLOCK_K = 64 if Kcap <= 64 else 128
    y = torch.empty_like(q_flat, dtype=torch.float32)
    grid = (NH,)
    _fc_single_q_attn_kernel[grid](
        q_flat, k_flat, v_flat, bias_flat, y, scale, Dh, Kcap,
        BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K, num_warps=4, num_stages=2
    )
    return y


@triton.jit
def _fc_fused_cache_kernel(
    # Query and biases
    Q, PBU, PBV,                    # Q: [T, H, Dh], PBU/PBV: [H, Dh]
    # KV cache base and strides for [B, O, H, Dh]
    KCACHE, VCACHE,
    stride_k_b, stride_k_o, stride_k_h, stride_k_d,
    # Metadata tensors
    BLOCK_TABLE,                    # [R, Wb]
    stride_bt_r, stride_bt_c,       # row/col stride
    QUERY_START_LOC,                # [R]
    DECODE_OFFSET,                  # [R]
    DOC_IDS,                        # [T]
    # Relative position projection for bias: REL[h, d, m], m in [0..2W]
    REL,
    stride_rel_h, stride_rel_d, stride_rel_m,
    # Output
    OUT,                            # [T, H, Dh]
    stride_q_t, stride_q_h, stride_q_d,   # strides for Q
    stride_o_t, stride_o_h, stride_o_d,   # strides for OUT
    pbu_s_h, pbu_s_d, pbv_s_h, pbv_s_d,  # strides for PBU/PBV
    # Scalar args
    T_live: tl.constexpr,           # not used at compile-time; grid controls bounds
    H_heads: tl.constexpr,
    Dh: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BT_WIDTH: tl.constexpr,
    W: tl.constexpr,                # 71
    KCAP: tl.constexpr,             # 72
    TWO_WP1: tl.constexpr,          # 143
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)   # token idx in [0, N_live)
    pid_h = tl.program_id(1)   # head idx in [0, H)
    if pid_h >= H_heads:
        return

    # Load req_id, q_start, dec_off
    req_id = tl.load(DOC_IDS + pid_n).to(tl.int32)
    q_start = tl.load(QUERY_START_LOC + req_id).to(tl.int32)
    dec_off = tl.load(DECODE_OFFSET + req_id).to(tl.int32)

    idx = tl.full((1,), pid_n, dtype=tl.int32)
    local_q = idx - q_start
    logical_q = local_q + dec_off
    start = tl.maximum(logical_q - W, 0)
    L = logical_q - start + 1
    pad = KCAP - L

    start_block = start // BLOCK_SIZE
    end_block = logical_q // BLOCK_SIZE
    start_off = start - start_block * BLOCK_SIZE

    # Block table lookups
    bt_row = BLOCK_TABLE + req_id * stride_bt_r
    i32_neg1 = tl.full((1,), -1, tl.int32)
    phys_start = tl.load(bt_row + start_block * stride_bt_c,
                     mask=start_block < BT_WIDTH, other=i32_neg1)
    phys_end   = tl.load(bt_row + end_block   * stride_bt_c,
                     mask=end_block   < BT_WIDTH, other=i32_neg1)
    two_block = start_block != end_block
    has_start = phys_start >= 0
    has_end = phys_end >= 0

    # Pointers for Q and biases
    d_offsets = tl.arange(0, BLOCK_D)
    q_row = Q + pid_n * stride_q_t + pid_h * stride_q_h
    pbu_row = PBU + pid_h * pbu_s_h
    pbv_row = PBV + pid_h * pbv_s_h

    # Load q_u, q_v in tiles and keep in registers
    # We will re-load per tile when needed

    # Streaming softmax variables and accumulator
    m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Iterate over slots in tiles of BLOCK_K
    for ko in range(0, KCAP, BLOCK_K):
        k_offsets = ko + tl.arange(0, BLOCK_K)
        relpos = k_offsets - pad
        thresh = BLOCK_SIZE - start_off
        use_end = (two_block & (relpos >= thresh))
        kv_block = tl.where(use_end, phys_end, phys_start)
        kv_off = tl.where(use_end, relpos - thresh, start_off + relpos)
        kv_off = tl.maximum(tl.minimum(kv_off, BLOCK_SIZE - 1), 0)
        valid = (k_offsets >= pad) & tl.where(use_end, has_end, has_start)

        # Compute QK scores for this tile across Dh in BLOCK_D chunks
        s_tile = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for do in range(0, Dh, BLOCK_D):
            d_idx = do + d_offsets
            d_mask = d_idx < Dh
            # q_u = q + pos_bias_u
            q_u = tl.load(q_row + d_idx * stride_q_d, mask=d_mask, other=0.0).to(tl.float32) + \
                  tl.load(pbu_row + d_idx * pbu_s_d, mask=d_mask, other=0.0).to(tl.float32)

            # K block gather pointers using strides
            k_ptrs = KCACHE \
                + kv_block[:, None] * stride_k_b \
                + kv_off[:, None] * stride_k_o \
                + pid_h * stride_k_h \
                + d_idx[None, :] * stride_k_d
            k_mask = (k_offsets[:, None] < KCAP) & d_mask[None, :] & valid[:, None]
            k_block = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BK, Bd]

            # s += q_u · k
            s_tile += tl.sum(k_block * q_u[None, :], axis=1)

        s_tile = s_tile * (1.0 / tl.sqrt(tl.full((1,), float(Dh), dtype=tl.float32)))

        # Add bias per slot: compute band_idx and per-slot dot(q_v, rel[:, m])
        # band_idx = clamp(W + (L - 1) - relpos, 0, 2W)
        band_idx = W + (L - 1) - relpos
        band_idx = tl.maximum(tl.minimum(band_idx, TWO_WP1 - 1), 0)

        bias_tile = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for do in range(0, Dh, BLOCK_D):
            d_idx = do + d_offsets
            d_mask = d_idx < Dh
            q_v = tl.load(q_row + d_idx * stride_q_d, mask=d_mask, other=0.0).to(tl.float32) + \
                  tl.load(pbv_row + d_idx * pbv_s_d, mask=d_mask, other=0.0).to(tl.float32)
            # Vectorized load of REL[h, d, m] for all m in this tile (band_idx)
            rel_ptrs = REL \
                + pid_h * stride_rel_h \
                + d_idx[:, None] * stride_rel_d \
                + band_idx[None, :] * stride_rel_m
            rel_mask = d_mask[:, None] & (band_idx[None, :] >= 0) & (band_idx[None, :] < TWO_WP1)
            rel_block = tl.load(rel_ptrs, mask=rel_mask, other=0.0).to(tl.float32)  # [Bd, BK]
            # Accumulate per-slot bias: sum_d q_v[d] * rel[h, d, m]
            bias_tile += tl.sum(rel_block * q_v[:, None], axis=0)
        bias_tile = bias_tile * (1.0 / tl.sqrt(tl.full((1,), float(Dh), dtype=tl.float32)))

        s_total = s_tile + bias_tile
        neg_inf = -float("inf")
        s_for_max = tl.where(valid, s_total, neg_inf)

        m_ij = tl.maximum(m_i, tl.max(s_for_max, axis=0))
        p = tl.where(valid, tl.exp(s_total - m_ij), 0.0)

        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p, axis=0)

        tile_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for do in range(0, Dh, BLOCK_D):
            d_idx = do + d_offsets
            d_mask = d_idx < Dh
            v_ptrs = VCACHE \
                + kv_block[:, None] * stride_k_b \
                + kv_off[:, None] * stride_k_o \
                + pid_h * stride_k_h \
                + d_idx[None, :] * stride_k_d
            v_mask = (k_offsets[:, None] < KCAP) & d_mask[None, :] & valid[:, None]
            v_block = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)  # [BK, Bd]
            # sum over BK -> Bd
            tile_v += tl.sum(p[:, None] * v_block, axis=0)

        acc = acc * alpha + tile_v

        m_i = m_ij
        l_i = l_ij

    out_tile = acc / l_i
    out_row = OUT + pid_n * stride_o_t + pid_h * stride_o_h
    tl.store(out_row + d_offsets * stride_o_d, out_tile, mask=d_offsets < Dh)


def _launch_fc_fused_cache_triton(
    query: torch.Tensor,            # [T, H, Dh]
    key_cache: torch.Tensor,        # [B, O, H, Dh]
    value_cache: torch.Tensor,      # [B, O, H, Dh]
    attn_metadata: FastConformerRPEMetadata,
    pos_bias_u: torch.Tensor,       # [H, Dh]
    pos_bias_v: torch.Tensor,       # [H, Dh]
    rel: torch.Tensor,              # [H, Dh, 2W+1]
    output: torch.Tensor,           # [T, H, Dh]
) -> None:
    T = int(attn_metadata.num_actual_tokens)
    H = query.shape[1]
    Dh = query.shape[2]
    block_size = int(attn_metadata.block_size)
    bt_width = attn_metadata.block_table.shape[1]
    W = 71
    Kcap = W + 1
    TWO_WP1 = 2 * W + 1

    bt_i32 = attn_metadata.block_table.to(torch.int32).contiguous()
    qstart_i32 = attn_metadata.query_start_loc.to(torch.int32).contiguous()
    decoff_i32 = attn_metadata.decode_offset.to(torch.int32).contiguous()
    docids_i32 = attn_metadata.doc_ids.to(torch.int32).contiguous()

    # Strides (assume contiguous last dim)
    stride_q_t = query.stride(0)
    stride_q_h = query.stride(1)
    stride_q_d = query.stride(2)
    stride_o_t = output.stride(0)
    stride_o_h = output.stride(1)
    stride_o_d = output.stride(2)
    stride_k_b = key_cache.stride(0)
    stride_k_o = key_cache.stride(1)
    stride_k_h = key_cache.stride(2)
    stride_k_d = key_cache.stride(3)
    pbu_s_h, pbu_s_d = pos_bias_u.stride(0), pos_bias_u.stride(1)
    pbv_s_h, pbv_s_d = pos_bias_v.stride(0), pos_bias_v.stride(1)
    # Block table strides
    stride_bt_r = bt_i32.stride(0)
    stride_bt_c = bt_i32.stride(1)
    # REL strides
    stride_rel_h = rel.stride(0)
    stride_rel_d = rel.stride(1)
    stride_rel_m = rel.stride(2)

    BLOCK_D = 128 if Dh > 64 else 64
    BLOCK_K = 64 if Kcap <= 64 else 128

    grid = (T, H)
    _fc_fused_cache_kernel[grid](
        query, pos_bias_u, pos_bias_v,
        key_cache, value_cache,
        stride_k_b, stride_k_o, stride_k_h, stride_k_d,
        bt_i32,
        stride_bt_r, stride_bt_c,
        qstart_i32,
        decoff_i32,
        docids_i32,
        rel,
        stride_rel_h, stride_rel_d, stride_rel_m,
        output,
        stride_q_t, stride_q_h, stride_q_d,
        stride_o_t, stride_o_h, stride_o_d,
        pbu_s_h, pbu_s_d,
        pbv_s_h, pbv_s_d,
        T_live=T, H_heads=H, Dh=Dh,
        BLOCK_SIZE=block_size, BT_WIDTH=bt_width,
        W=W, KCAP=Kcap, TWO_WP1=TWO_WP1,
        BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

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
        return self._forward_v4(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

    def _forward_v1(
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
    
    def _forward_v4(
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
        Fully fused Triton path: compute metadata and attention inside the kernel.
        Assumes fixed window W=71 (K_cap=72) and head equality (H==num_kv_heads).
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Output scaling not supported for FastConformer RPE.")
        if attn_metadata is None:
            return output
        if not _HAS_TRITON:
            # Fallback if Triton isn't present
            return self._forward_v2(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

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

        N_live = int(attn_metadata.num_actual_tokens)
        if N_live == 0:
            output.zero_()
            return output

        # Precompute rel projection tensor [H, Dh, 2W+1] once (outside the kernel)
        W = 71
        rel = self._get_rel_proj(query.device, query.dtype, W)

        _launch_fc_fused_cache_triton(
            query.contiguous(),                      # use raw Q; kernel adds pos biases
            key_cache,
            value_cache,
            attn_metadata,
            self.pos_bias_u.contiguous(),
            self.pos_bias_v.contiguous(),
            rel.contiguous(),
            output,
        )

        # Zero remaining tokens beyond live
        if output.shape[0] > N_live:
            output[N_live:].zero_()
        return output

    def _forward_v3(
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
        Triton path: fused single-query attention over a small window K_cap = W+1.
        Leaves _forward_v2 untouched; can be toggled by the caller.
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
        bt_per_req = bt.index_select(0, req_ids)
        phys_start = bt_per_req.gather(1, start_block.unsqueeze(1)).squeeze(1)
        phys_end = bt_per_req.gather(1, end_block.unsqueeze(1)).squeeze(1)

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

        flat_key_cache = key_cache.view(-1, H, Dh)      # [(num_blocks*block_size), H, Dh]
        flat_value_cache = value_cache.view(-1, H, Dh)  # [(num_blocks*block_size), H, Dh]
        linear_idx = kv_block * block_size + kv_off     # [N, K_cap]
        flat_idx = linear_idx.reshape(-1)
        ks = flat_key_cache.index_select(0, flat_idx).view(N_live, K_cap, H, Dh)    # [N, K_cap, H, Dh]
        vs = flat_value_cache.index_select(0, flat_idx).view(N_live, K_cap, H, Dh)  # [N, K_cap, H, Dh]

        rel = self._get_rel_proj(device, q_u.dtype, W)  # [H, Dh, 2W+1]
        band = torch.einsum("n h d, h d m -> n h m", q_v[:N_live], rel) * (Dh ** -0.5)
        band_idx = (W + (L.unsqueeze(1) - 1) - relpos).clamp(0, 2 * W)
        bias_g = band.gather(2, band_idx.unsqueeze(1).expand(-1, H, -1))
        bias_g = bias_g.masked_fill(~valid.unsqueeze(1), float("-inf"))  # [N, H, K_cap]

        NH = N_live * H
        q_flat = q_u[:N_live].reshape(NH, Dh).contiguous()                                # [NH, Dh]
        k_flat = ks.permute(0, 2, 3, 1).reshape(NH, Dh, K_cap).contiguous()               # [NH, Dh, K_cap]
        v_flat = vs.permute(0, 2, 1, 3).reshape(NH, K_cap, Dh).contiguous()               # [NH, K_cap, Dh]
        bias_flat = bias_g.reshape(NH, K_cap).contiguous()                                # [NH, K_cap]

        y_flat = _launch_fc_single_q_attn_triton(q_flat, k_flat, v_flat, bias_flat, float(Dh ** -0.5))
        y = y_flat.view(N_live, H, Dh).to(q_u.dtype)

        output.zero_()
        output[:N_live].copy_(y)
        return output

    def _forward_v2(
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
        bt_per_req = bt.index_select(0, req_ids)
        phys_start = bt_per_req.gather(1, start_block.unsqueeze(1)).squeeze(1)
        phys_end = bt_per_req.gather(1, end_block.unsqueeze(1)).squeeze(1)

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

        flat_key_cache = key_cache.view(-1, H, Dh)      # [(num_blocks*block_size), H, Dh]
        flat_value_cache = value_cache.view(-1, H, Dh)  # [(num_blocks*block_size), H, Dh]
        linear_idx = kv_block * block_size + kv_off     # [N, K_cap]
        flat_idx = linear_idx.reshape(-1)
        ks = flat_key_cache.index_select(0, flat_idx).view(N_live, K_cap, H, Dh)    # [N, K_cap, H, Dh]
        vs = flat_value_cache.index_select(0, flat_idx).view(N_live, K_cap, H, Dh)  # [N, K_cap, H, Dh]

        rel = self._get_rel_proj(device, q_u.dtype, W)  # [H, Dh, 2W+1]
        band = torch.einsum("n h d, h d m -> n h m", q_v[:N_live], rel) * (Dh ** -0.5)
        band_idx = (W + (L.unsqueeze(1) - 1) - relpos).clamp(0, 2 * W)
        bias_g = band.gather(2, band_idx.unsqueeze(1).expand(-1, H, -1))
        bias_g = bias_g.masked_fill(~valid.unsqueeze(1), float("-inf"))  # [N, H, K_cap]
        
        q_flat = q_u[:N_live].reshape(N_live * H, Dh).contiguous()                                # [N*H, Dh]
        k_flat = ks.permute(0, 2, 3, 1).reshape(N_live * H, Dh, K_cap).contiguous()               # [N*H, Dh, K_cap]
        v_flat = vs.permute(0, 2, 1, 3).reshape(N_live * H, K_cap, Dh).contiguous()               # [N*H, K_cap, Dh]
        bias_flat = bias_g.reshape(N_live * H, K_cap).contiguous()                                # [N*H, K_cap]
        
        scores = torch.bmm(q_flat.unsqueeze(1), k_flat).squeeze(1) * (Dh ** -0.5)                 # [N*H, K_cap]
        scores = scores + bias_flat
        attn = torch.softmax(scores, dim=-1)                                                      # [N*H, K_cap]
        y = torch.bmm(attn.unsqueeze(1), v_flat).squeeze(1).view(N_live, H, Dh)                   # [N, H, Dh]
        
        output.zero_()
        output[:N_live].copy_(y)
        return output
