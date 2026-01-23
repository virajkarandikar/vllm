#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar, Optional
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from vllm.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
reshape_and_cache_flash = triton_reshape_and_cache_flash

from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
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
    Q, PBU, PBV,
    KCACHE, VCACHE,
    stride_k_b, stride_k_o, stride_k_h, stride_k_d,
    BLOCK_TABLE,
    stride_bt_r, stride_bt_c,
    QUERY_START_LOC,
    DECODE_OFFSET,
    DOC_IDS,
    REL,
    stride_rel_h, stride_rel_d, stride_rel_m,
    OUT,
    stride_q_t, stride_q_h, stride_q_d,
    stride_o_t, stride_o_h, stride_o_d,
    pbu_s_h, pbu_s_d, pbv_s_h, pbv_s_d,
    T_live: tl.constexpr,
    H_heads: tl.constexpr,
    Dh: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BT_WIDTH: tl.constexpr,
    W: tl.constexpr,
    KCAP: tl.constexpr,
    TWO_WP1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_h >= H_heads:
        return

    req_id = tl.load(DOC_IDS + pid_n).to(tl.int32)
    q_start = tl.load(QUERY_START_LOC + req_id).to(tl.int32)
    dec_off = tl.load(DECODE_OFFSET + req_id).to(tl.int32)

    idx = tl.full((1,), pid_n, dtype=tl.int32)
    logical_q = idx - q_start + dec_off
    start = tl.maximum(logical_q - W, 0)
    L = logical_q - start + 1
    pad = KCAP - L

    start_block = start // BLOCK_SIZE
    end_block = logical_q // BLOCK_SIZE
    start_off = start - start_block * BLOCK_SIZE

    bt_row = BLOCK_TABLE + req_id * stride_bt_r
    i32_neg1 = tl.full((1,), -1, tl.int32)
    phys_start = tl.load(bt_row + start_block * stride_bt_c,
                         mask=start_block < BT_WIDTH,
                         other=i32_neg1)
    phys_end   = tl.load(bt_row + end_block   * stride_bt_c,
                         mask=end_block   < BT_WIDTH,
                         other=i32_neg1)

    two_block = start_block != end_block
    has_start = phys_start >= 0
    has_end   = phys_end >= 0

    k_offsets = tl.arange(0, BLOCK_K)
    relpos = k_offsets - pad
    thresh = BLOCK_SIZE - start_off

    use_end = two_block & (relpos >= thresh)
    kv_block = tl.where(use_end, phys_end, phys_start)
    kv_off   = tl.where(use_end, relpos - thresh, start_off + relpos)
    kv_off   = tl.maximum(tl.minimum(kv_off, BLOCK_SIZE - 1), 0)

    valid = (k_offsets >= pad) & tl.where(use_end, has_end, has_start)

    safe_block = tl.maximum(phys_start, 0)
    safe_off   = tl.full((BLOCK_K,), 0, tl.int32)
    kv_block = tl.where(valid, kv_block, safe_block)
    kv_off   = tl.where(valid, kv_off,   safe_off)

    mask_k = k_offsets < KCAP

    d_offsets = tl.arange(0, BLOCK_D)
    q_row = Q + pid_n * stride_q_t + pid_h * stride_q_h
    pbu_row = PBU + pid_h * pbu_s_h
    pbv_row = PBV + pid_h * pbv_s_h

    logits = tl.full((BLOCK_K,), -float("inf"), dtype=tl.float32)

    s_tile = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for do in range(0, Dh, BLOCK_D):
        d_idx = do + d_offsets
        d_mask = d_idx < Dh

        q_u = tl.load(q_row + d_idx * stride_q_d,
                      mask=d_mask, other=0.0).to(tl.float32) \
            + tl.load(pbu_row + d_idx * pbu_s_d,
                      mask=d_mask, other=0.0).to(tl.float32)

        k_ptrs = KCACHE \
            + kv_block[:, None] * stride_k_b \
            + kv_off[:, None]   * stride_k_o \
            + pid_h * stride_k_h \
            + d_idx[None, :] * stride_k_d
        k_mask = mask_k[:, None] & d_mask[None, :] & valid[:, None]
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        s_tile += tl.sum(k_block * q_u[None, :], axis=1)

    bias_tile = tl.zeros((BLOCK_K,), dtype=tl.float32)
    band_idx = W + (L - 1) - relpos
    band_idx = tl.maximum(tl.minimum(band_idx, TWO_WP1 - 1), 0)

    for do in range(0, Dh, BLOCK_D):
        d_idx = do + d_offsets
        d_mask = d_idx < Dh

        q_v = tl.load(q_row + d_idx * stride_q_d,
                      mask=d_mask, other=0.0).to(tl.float32) \
            + tl.load(pbv_row + d_idx * pbv_s_d,
                      mask=d_mask, other=0.0).to(tl.float32)

        rel_ptrs = REL \
            + pid_h * stride_rel_h \
            + d_idx[:, None] * stride_rel_d \
            + band_idx[None, :] * stride_rel_m
        rel_mask = d_mask[:, None] & mask_k[None, :]
        rel_block = tl.load(rel_ptrs, mask=rel_mask, other=0.0).to(tl.float32)

        bias_tile += tl.sum(rel_block * q_v[:, None], axis=0)

    inv_sqrt_d = 1.0 / tl.sqrt(tl.full((1,), float(Dh), dtype=tl.float32))
    logits = (s_tile + bias_tile) * inv_sqrt_d

    neg_inf = -float("inf")
    logits = tl.where(valid & mask_k, logits, neg_inf)

    max_l = tl.max(logits, axis=0)
    exp_l = tl.exp(logits - max_l)
    exp_l = tl.where(valid & mask_k, exp_l, 0.0)
    denom = tl.sum(exp_l, axis=0)
    p = exp_l / denom

    out_tile = tl.zeros((BLOCK_D,), dtype=tl.float32)
    out_row = OUT + pid_n * stride_o_t + pid_h * stride_o_h

    for do in range(0, Dh, BLOCK_D):
        d_idx = do + d_offsets
        d_mask = d_idx < Dh

        v_ptrs = VCACHE \
            + kv_block[:, None] * stride_k_b \
            + kv_off[:, None]   * stride_k_o \
            + pid_h * stride_k_h \
            + d_idx[None, :] * stride_k_d
        v_mask = mask_k[:, None] & d_mask[None, :] & valid[:, None]
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # (BLOCK_K, Dh_chunk) * (BLOCK_K,) -> Dh_chunk
        contrib = tl.sum(v_block * p[:, None], axis=0)
        out_tile = tl.where(d_mask, contrib, out_tile)

        tl.store(out_row + d_idx * stride_o_d,
                 out_tile.to(tl.float32),
                 mask=d_mask)


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
    W = 70
    Kcap = W + 1
    TWO_WP1 = 2 * W + 1

    bt_i32 = attn_metadata.block_table.to(torch.int32).contiguous()
    qstart_i32 = attn_metadata.query_start_loc.to(torch.int32).contiguous()
    decoff_i32 = attn_metadata.decode_offset.to(torch.int32).contiguous()
    docids_i32 = attn_metadata.doc_ids.to(torch.int32).contiguous()

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
    stride_bt_r = bt_i32.stride(0)
    stride_bt_c = bt_i32.stride(1)
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
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

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
        # NOTE(vklimkov): clean up possible attention implementations for RPE.
        # see git history for more details.
        assert output is not None
        if attn_metadata is None:
            return output

        key_cache, value_cache = kv_cache.unbind(0)
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
        W = 70
        rel = self._get_rel_proj(query.device, query.dtype, W)

        _launch_fc_fused_cache_triton(
            query.contiguous(),
            key_cache,
            value_cache,
            attn_metadata,
            self.pos_bias_u.contiguous(),
            self.pos_bias_v.contiguous(),
            rel.contiguous(),
            output,
        )
        if output.shape[0] > N_live:
            output[N_live:].zero_()

        return output
