# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.attention.layer import Attention
from vllm.sequence import IntermediateTensors

from vllm.v1.attention.backends.fastconformer_attn import (
    FastConformerBackend,
    FastConformerMetadata,
)
from vllm.forward_context import get_forward_context
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend, FlexAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheSpec, FastConformerConvSpec
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
)

from vllm.transformers_utils.configs.fastconformer import FastConformerCTCConfig
import math

try:
    from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
    torch_sdpa = torch.compile(torch_sdpa, fullgraph=True)
except ImportError:
    torch_sdpa = None

def _is_dummy_run() -> bool:
    attn_metadata = get_forward_context().attn_metadata
    return attn_metadata is None


def _check(save_pth: str, x: torch.Tensor):
    if _is_dummy_run():
        return
    print(f"[vllm_debug] checking {save_pth}...")
    ref_t = torch.load(save_pth)
    ref_t = ref_t[:,:,:]
    print(f"[vllm_debug] x.shape {x.shape} ref_t.shape {ref_t.shape} x.dtype {x.dtype} ref_t.dtype {ref_t.dtype}")
    a = x.detach().float().cpu()
    b = ref_t.detach().float().cpu()
    diff = (a-b).abs()
    rel = diff / (b.abs() + 1e-6)
    print(f"[vllm_debug] shape={tuple(a.shape)} | mean abs diff={diff.mean():.6f} | max abs diff={diff.max():.6f} | mean rel diff={rel.mean():.6f}")

def _dbg_save(save_pth: str, x: torch.Tensor):
    if _is_dummy_run():
        return
    print(f"[vllm_debug] saving {save_pth}...")
    torch.save(x, save_pth)
    print(f"[vllm_debug] saved {save_pth}")

# TODO: This module needs to be cleaned up. It is designed to replicate the NeMo subsampler
# in a lighter weight manner, but it is currently unclear which configuration of the subsampler
# is appropriate for inference (i.e. which matches the training configuration). For the time being,
# the main FastConformerCTC modelling class imports the NeMo subsampler and uses it directly.
class NemoSubsample8x2D(nn.Module):
    def __init__(self, d_out: int, mid_ch: int, mels: int):
        super().__init__()
        self.mels = mels
        self.mid_ch = mid_ch
        self.act = nn.ReLU()

        self.pad = nn.ConstantPad2d((2, 1, 2, 1), 0)

        self.conv0 = nn.Conv2d(1, self.mid_ch, kernel_size=3, stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=3, stride=(2, 2),
                               padding=0, groups=self.mid_ch)
        self.conv3 = nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=1, stride=1, padding=0)

        self.conv5 = nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=3, stride=(2, 2),
                               padding=0, groups=self.mid_ch)
        self.conv6 = nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=1, stride=1, padding=0)

        self.out = nn.Linear(self.mid_ch * 11, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape
        assert F == self.mels, f"Expected mel dim {self.mels}, got {F}"

        y = x.view(B, 1, T, F)
        y = self.act(self.conv0(self.pad(y)))     # stage 0

        y = self.conv2(self.pad(y))               # depthwise
        y = self.act(self.conv3(y))               # pointwise + act

        y = self.conv5(self.pad(y))               # depthwise
        y = self.act(self.conv6(y))               # pointwise + act

        B, C, T8, Fp = y.shape
        if C * Fp != 2816:  # should be 256 * 11
            raise RuntimeError(f"Subsampler produced C*F'={C}*{Fp}={C*Fp}, expected 2816.")

        y = y.permute(0, 2, 1, 3).contiguous().view(B, T8, C * Fp)  # [B, T/8, 256*11]
        y = self.out(y)                                            # [B, T/8, d_out]
        return y

class ConformerFFN(nn.Module):
    """Conformer FeedForward module."""
    def __init__(self, d_model: int, ff_mult: int = 4):
        super().__init__()
        d_ff = ff_mult * d_model
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class FastConformerConvCache(torch.nn.Module, AttentionLayerBase):
    def __init__(self, d_model: int, k: int, prefix: str, cache_config: CacheConfig, dtype: torch.dtype):
        super().__init__()
        self.d_model = int(d_model)
        self.k = int(k)
        self.left_ctx = self.k - 1  # L

        # The conv cache page size must match the attn page size by vLLM requirements.
        # TODO: is there a less hacky way to do this?
        self.left_shape = self.left_ctx * 32

        self.prefix = prefix
        self.cache_config = cache_config
        self.kv_cache = [torch.tensor([])]
        self.dtype = dtype

        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            block_size=self.cache_config.block_size,
            shape=(self.left_shape, self.d_model),
            dtype=self.dtype,
        )

    def forward(self):
        pass

    def get_attn_backend(self) -> AttentionBackend:
        return FastConformerBackend

class _TransformedShawBias:
    def __init__(self, W: int):
        self.W = int(W)
        self.band_bias = None           # [B,H,T,2W+1]
        self.doc_ids = None             # [num_q]
        self.decode_offset = None       # [num_reqs]
        self.physical_to_logical = None # [num_reqs, total_blocks]
        self.block_size = None          # int

    def bind(self, attn_meta, band_bias: torch.Tensor):
        self.band_bias = band_bias
        self.doc_ids = attn_meta.doc_ids
        self.decode_offset = attn_meta.decode_offset
        self.physical_to_logical = attn_meta.physical_to_logical
        self.block_size = int(attn_meta.block_size)

    # @torch.jit.script_if_tracing
    def __call__(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        # block_sz = self.block_size
        # physical_kv_block = torch.div(physical_kv_idx, block_sz, rounding_mode='floor')
        # physical_kv_offset = physical_kv_idx % block_sz

        # doc = self.doc_ids[q_idx.long()]
        # logical_block_idx = self.physical_to_logical[doc.long(), physical_kv_block.long()]
        # logical_kv_idx = logical_block_idx * block_sz + physical_kv_offset

        # live_block = logical_block_idx >= 0
        # within_lower = logical_kv_idx >= 0
        # is_valid = live_block & within_lower

        # logical_q_idx = q_idx + self.decode_offset[doc.long()]
        # rel = torch.clamp(logical_q_idx - logical_kv_idx, -self.W, self.W) + self.W
        # local_q = logical_q_idx - self.decode_offset[doc.long()]

        # bias = self.band_bias[doc.long(), h.long(), local_q.long(), rel.long()]
        # return torch.where(is_valid, score + bias, score)
        return score

class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window: int,
                 cache_config: CacheConfig, prefix: str):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.window = int(window)
        self.prefix = prefix

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.dh))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.dh))

        self.attn = Attention(
            num_heads=self.h,
            head_size=self.dh,
            scale=self.dh ** -0.5,
            num_kv_heads=self.h,
            cache_config=CacheConfig(
                sliding_window=self.window,
                cache_dtype="auto",
                block_size=cache_config.block_size,
                calculate_kv_scales=False,
            ),
            prefix=self.prefix,
            attn_backend=FlexAttentionBackend,
        )

        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        if cache_config.block_size != 128:
            raise Exception(
                f"attn cache block size must be 128, got {cache_config.block_size}"
            )

        self._rel_cache: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

        self._shaw_mod = _TransformedShawBias(self.window)
        self._blockmask_cache: dict[tuple[int, int, int, int, int], object] = {}

        self.register_buffer(
            "qkv_weight",
            torch.empty(3 * d_model, d_model, dtype=self.q_proj.weight.dtype)
        )
        self.register_buffer(
            "qkv_bias",
            torch.empty(3 * d_model, dtype=self.q_proj.weight.dtype)
        )
        self._qkv_fused_ready: bool = False

    def _fused_qkv_projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._qkv_fused_ready:
            qkv = F.linear(x, self.qkv_weight, self.qkv_bias)  # [B, T, 3D]
        else:
            raise Exception("this should not happen")
        D = x.size(-1)
        q, k, v = qkv.split(D, dim=-1)
        return q, k, v

    def _forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh
        assert D == H * Dh

        q, k, v = self._fused_qkv_projection(x)
        q = q.view(B, T, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B, T, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, T, H, Dh).transpose(1, 2).contiguous()

        device, idtype = x.device, x.dtype
        pos_idx = torch.arange(T - 1, -T, -1, device=device)[:T]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / D))
        sin = torch.sin(pos_idx[:, None].to(torch.float32) * div[None, :])
        cos = torch.cos(pos_idx[:, None].to(torch.float32) * div[None, :])
        pos = torch.zeros(T, D, device=device, dtype=torch.float32)
        pos[:, 0::2] = sin
        pos[:, 1::2] = cos

        p = self.linear_pos(pos.to(idtype)).view(T, H, Dh).permute(1, 0, 2).contiguous()
        p = p.unsqueeze(0).expand(B, -1, -1, -1)

        q_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        scores_ac = torch.matmul(q_u, k.transpose(-2, -1))  # [B,H,T,T]

        raw_bd = torch.matmul(q_v, p.transpose(-2, -1))     # [B,H,T,T]
        b, h, qlen, pos_len = raw_bd.size()
        bd = F.pad(raw_bd, (1, 0))
        bd = bd.view(b, h, pos_len + 1, qlen)[:, :, 1:].view(b, h, qlen, pos_len)
        scores_bd = bd

        scores = (scores_ac + scores_bd) * (Dh ** -0.5)

        causal = torch.ones(T, T, device=device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal.view(1, 1, T, T), float("-inf"))

        W = int(self.window)
        if W > 0 and W < T:
            idx = torch.arange(T, device=device)
            q_idx = idx.view(1, 1, T, 1)
            k_idx = idx.view(1, 1, 1, T)
            allowed = (k_idx <= q_idx) & ((q_idx - k_idx) <= W)
            scores = scores.masked_fill(~allowed, float("-inf"))

        scores32 = scores.to(torch.float32)
        scores32 = scores32 - torch.amax(scores32, dim=-1, keepdim=True)
        probs = torch.softmax(scores32, dim=-1).to(idtype)

        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(ctx)

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

    def _build_shaw_band_bias(self, q: torch.Tensor, W: int) -> torch.Tensor:
        B, H, T, Dh = q.shape
        device, dtype = q.device, q.dtype
        rel = self._get_rel_proj(device, dtype, W)  # [H, Dh, 2W+1]

        q_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)   # [B,H,T,Dh]
        band_bias = torch.einsum("bhtd,hdm->bhtm", q_v, rel)  # [B,H,T,2W+1]
        band_bias = band_bias * (Dh ** -0.5)
        return band_bias

    def _mask_cache_key(self, meta: FlexAttentionMetadata) -> tuple[int, int, int, int]:
        return (int(meta.block_table.data_ptr()),
                int(meta.q_block_size),
                int(meta.kv_block_size),
                int(self.window))

    def _install_exact_mask_cached(self, attn_meta: FlexAttentionMetadata):
        if getattr(attn_meta, "_installed_exact_mask", False):
            return

        W = int(self.window)
        def logical_mask_mod(b: torch.Tensor,
                             h: torch.Tensor,
                             q_idx: torch.Tensor,
                             kv_idx: torch.Tensor) -> torch.Tensor:
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) <= W)

        attn_meta.sliding_window = None
        attn_meta.logical_mask_mod = logical_mask_mod
        attn_meta.direct_build = True

        key = self._mask_cache_key(attn_meta)
        bm = self._blockmask_cache.get(key)
        print(f"[vllm_debug] id(bm): {id(bm)}")
        if bm is None:
            attn_meta.mask_mod = attn_meta.get_mask_mod()
            bm = attn_meta.build_block_mask()
            self._blockmask_cache[key] = bm
        attn_meta.block_mask = bm
        attn_meta._installed_exact_mask = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_sdpa_2(x)

    def _forward_sdpa(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        q, k, v = self._fused_qkv_projection(x)
        q = q.view(B, T, H, Dh)
        k = k.view(B, T, H, Dh)
        v = v.view(B, T, H, Dh)

        q = q.permute(0, 2, 1, 3).reshape(B * H, T, Dh)
        k = k.permute(0, 2, 1, 3).reshape(B * H, T, Dh)
        v = v.permute(0, 2, 1, 3).reshape(B * H, T, Dh)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_output = attn_output.view(B, H, T, Dh).permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.o_proj(attn_output)
        return out

    def _forward_sdpa_2(self, x: torch.Tensor) -> torch.Tensor:
        device, idtype = x.device, x.dtype
        H, Dh = self.h, self.dh
        W = int(self.window)
        K_cap = W + 1
        D_model = H * Dh

        q_lin, k_lin, v_lin = self._fused_qkv_projection(x)

        if x.ndim == 3:
            B, T, D = x.shape
            assert D == D_model
            N_tokens = B * T
            q = q_lin.view(B, T, H, Dh).transpose(1, 2).reshape(N_tokens, H, Dh)
            k = k_lin.view(B, T, H, Dh).transpose(1, 2).reshape(N_tokens, H, Dh)
            v = v_lin.view(B, T, H, Dh).transpose(1, 2).reshape(N_tokens, H, Dh)
            reshape_kind = "bt"
            reshape_info = (B, T)
        elif x.ndim == 2:
            N_tokens, D = x.shape
            assert D == D_model
            q = q_lin.view(N_tokens, H, Dh)
            k = k_lin.view(N_tokens, H, Dh)
            v = v_lin.view(N_tokens, H, Dh)
            reshape_kind = "packed"
            reshape_info = None
        else:
            raise ValueError("x must be of shape [B, T, D] or [N, D].")

        # u/v position biases
        q_u = q + self.pos_bias_u.unsqueeze(0)  # [N, H, Dh]
        q_v = q + self.pos_bias_v.unsqueeze(0)  # [N, H, Dh]

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if not isinstance(attn_meta_all, dict):
            # fallback path during dummy run
            if x.ndim == 2:
                x_bt = x.view(1, N_tokens, D_model)
                return self._forward_ref(x_bt).view(N_tokens, D_model)
            return self._forward_ref(x)

        attn_meta: FlexAttentionMetadata = attn_meta_all[self.prefix]
        block_size = int(attn_meta.block_size)
        assert W < block_size, f"window ({W}) must be < cache block_size ({block_size})"

        N_live = int(attn_meta.num_actual_tokens)
        assert N_live <= q_u.shape[0]

        self_kv_cache = self.attn.kv_cache[fctx.virtual_engine]  # [2, num_blocks, block_size, H, Dh]
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            k, v,
            self_kv_cache[0], self_kv_cache[1],
            attn_meta.slot_mapping,
            self.attn.impl.kv_cache_dtype,
            self._k_scale, self._v_scale,
        )
        key_cache, value_cache = self_kv_cache.unbind(0)  # [num_blocks, block_size, H, Dh]

        req_ids = attn_meta.doc_ids[:N_live].to(torch.long)
        q_start = attn_meta.query_start_loc.index_select(0, req_ids)
        dec_off = attn_meta.decode_offset.index_select(0, req_ids)

        idx = torch.arange(N_live, device=device, dtype=torch.long)
        local_q = idx - q_start
        logical_q = local_q + dec_off

        start = torch.clamp(logical_q - W, min=0)
        L = logical_q - start + 1
        pad = K_cap - L

        start_block = torch.div(start, block_size, rounding_mode='floor')
        end_block   = torch.div(logical_q, block_size, rounding_mode='floor')
        start_off   = start - start_block * block_size
        end_off     = logical_q - end_block * block_size

        phys_start = attn_meta.block_table.index_select(0, req_ids).gather(1, start_block.unsqueeze(1)).squeeze(1)
        phys_end   = attn_meta.block_table.index_select(0, req_ids).gather(1, end_block.unsqueeze(1)).squeeze(1)

        two_block = (start_block != end_block)

        slot = torch.arange(K_cap, device=device, dtype=torch.long).unsqueeze(0).expand(N_live, K_cap)
        valid = slot >= pad.unsqueeze(1)
        relpos = slot - pad.unsqueeze(1)

        thresh = (block_size - start_off).unsqueeze(1)

        kv_block_start = phys_start.unsqueeze(1).expand_as(slot)
        kv_off_start   = start_off.unsqueeze(1) + relpos

        kv_block_end = phys_end.unsqueeze(1).expand_as(slot)
        kv_off_end   = relpos - thresh

        use_end = two_block.unsqueeze(1) & (relpos >= thresh)
        kv_block = torch.where(use_end, kv_block_end, kv_block_start)
        kv_off   = torch.where(use_end, kv_off_end,   kv_off_start)

        kv_block = kv_block.clamp_min(0)
        kv_off   = kv_off.clamp(min=0, max=block_size - 1)

        ks = key_cache[kv_block, kv_off]
        vs = value_cache[kv_block, kv_off]
        if not valid.all():
            mask4 = valid.unsqueeze(2).unsqueeze(3)
            ks = torch.where(mask4, ks, torch.zeros(1, dtype=ks.dtype, device=ks.device)).contiguous()
            vs = torch.where(mask4, vs, torch.zeros(1, dtype=vs.dtype, device=vs.device)).contiguous()

        rel = self._get_rel_proj(device, idtype, W)                              # [H, Dh, 2W+1]
        band = torch.einsum("n h d, h d m -> n h m", q_v[:N_live], rel) * (Dh ** -0.5)  # [N, H, 2W+1]
        band_idx = (W + (L.unsqueeze(1) - 1) - relpos).clamp(0, 2 * W)           # [N, K_cap]
        bias_g = band.gather(2, band_idx.unsqueeze(1).expand(-1, H, -1))
        neg_inf = torch.tensor(float("-inf"), dtype=idtype, device=device)
        bias_g = torch.where(valid.unsqueeze(1), bias_g, neg_inf)

        attn_bias_sdpa = bias_g.reshape(N_live * H, 1, K_cap).contiguous()

        q_sdpa = q_u[:N_live].reshape(N_live * H, 1, Dh)
        # ks/vs: [N, K, H, Dh] -> [N*H, K, Dh]
        k_sdpa = ks.permute(0, 2, 1, 3).reshape(N_live * H, K_cap, Dh)
        v_sdpa = vs.permute(0, 2, 1, 3).reshape(N_live * H, K_cap, Dh)

        assert torch_sdpa is not None

        y = torch_sdpa(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_bias_sdpa,
            dropout_p=0.0,
            is_causal=False,
        )  # [N*H, 1, Dh]
        y = y.view(N_live, H, Dh)

        out_buf = torch.zeros_like(q_u)          # [N_tokens, H, Dh]
        out_buf[:N_live] = y
        if reshape_kind == "bt":
            B, T = reshape_info
            attn_output = out_buf.view(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B, T, D_model)
        else:
            attn_output = out_buf.reshape(-1, D_model)

        return self.o_proj(attn_output)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        q, k, v = self._fused_qkv_projection(x)
        q = q.view(B, T, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B, T, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, T, H, Dh).transpose(1, 2).contiguous()

        q_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)

        W = int(self.window)
        # band_bias = self._build_shaw_band_bias(q, W)  # [B,H,T,2W+1]

        query = q_u.reshape(-1, H, Dh)
        key   = k.reshape(-1, H, Dh)
        value = v.reshape(-1, H, Dh)

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if not isinstance(attn_meta_all, dict):
            return self._forward_ref(x)

        attn_meta: FlexAttentionMetadata = attn_meta_all[self.prefix]

        self._install_exact_mask_cached(attn_meta)

        # self._shaw_mod.bind(attn_meta, band_bias)
        if not getattr(attn_meta, "_installed_shaw_mod", False):
            # attn_meta.transformed_score_mod = self._shaw_mod
            attn_meta.transformed_score_mod = None
            attn_meta._installed_shaw_mod = True

        self_kv_cache = self.attn.kv_cache[fctx.virtual_engine]
        out_buf = torch.empty_like(query)  # [num_tokens, H, Dh]
        out = self.attn.impl.forward(
            layer=self,
            query=query,
            key=key,
            value=value,
            kv_cache=self_kv_cache,
            attn_metadata=attn_meta,
            output=out_buf,
        )
        out = out.view(B, T, H, Dh).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, k: int, prefix: str, cache_config: CacheConfig, dtype: torch.dtype):
        super().__init__()
        assert k % 2 == 1
        self.prefix = prefix

        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw  = nn.Conv1d(d_model, d_model, k, padding=(k-1)//2, groups=d_model)
        self.bn  = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)

        self.conv_cache = FastConformerConvCache(
            d_model=d_model, k=k, prefix=f"{prefix}.conv_cache", cache_config=cache_config,
            dtype=dtype,
        )
        self.d_model = d_model

    def forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(0)

        B, T, D = x.shape
        assert D == self.d_model

        y = x.transpose(1, 2)         # [B, D, T]
        y = self.pw1(y)               # [B, 2D, T]
        a, b = y.chunk(2, dim=1)
        pre_dw = a * torch.sigmoid(b)  # GLU

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if not isinstance(attn_meta_all, dict):
            # dummy path: plain conv
            y_dw = self.dw(pre_dw)
            y_dw = self.bn(y_dw)
            y_dw = F.silu(y_dw)
            y = self.pw2(y_dw)
            out = y.transpose(1, 2)   # [B, T, D]
            if is_2d:
                out = out.squeeze(0)
            return out

        attn_metadata: FastConformerMetadata = attn_meta_all[self.conv_cache.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        # the length of the cache is not exactly equal to the left context length
        # because of an invariant that vLLM enforces where the page size must
        # be the same for both conv and attention layers.
        store = self.conv_cache.kv_cache[fctx.virtual_engine]  # [num_pages, L', D]
        L = self.conv_cache.left_ctx
        hist_full = store[page_indices]                   # [B, L', D]
        hist = hist_full[:, -L:, :].transpose(1, 2).contiguous()   # [B, D, L]

        x_cat = torch.cat([hist, pre_dw], dim=-1)                 # [B, D, L+T]

        y_dw = F.conv1d(
            x_cat,
            self.dw.weight,
            self.dw.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=D,
        )[:, :, -T:]
        y_dw = self.bn(y_dw)
        y_dw = F.silu(y_dw)
        y = self.pw2(y_dw).transpose(1, 2)                        # [B, T, D]

        # update cache
        new_hist = x_cat[:, :, -L:]                       # [B, D, L]
        store[page_indices, -L:, :] = new_hist.transpose(1, 2)   # [B, L, D]

        if is_2d:
            y = y.squeeze(0)
        return y

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "forward_cuda expects a 2D tensor [T, D]"
        T, D = x.shape
        assert D == self.d_model

        y = x.transpose(0, 1).unsqueeze(0)   # [1, D, T]
        y = self.pw1(y)                      # [1, 2D, T]
        a, b = y.chunk(2, dim=1)
        pre_dw = a * torch.sigmoid(b)        # [1, D, T]

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if not isinstance(attn_meta_all, dict):
            return self.forward_ref(x)

        attn_metadata: FastConformerMetadata = attn_meta_all[self.conv_cache.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        store = self.conv_cache.kv_cache[fctx.virtual_engine]      # [num_pages, L', D]
        conv_state = store.contiguous().transpose(1, 2)            # [num_pages, D, L']

        K = self.dw.weight.size(2)
        conv_weights = self.dw.weight.view(D, K)
        conv_bias = self.dw.bias

        pre_dw_2d = pre_dw.squeeze(0)             # [D, T]
        query_start_loc = attn_metadata.query_start_loc
        has_initial_state = torch.ones(
            page_indices.size(0), dtype=torch.bool, device=pre_dw_2d.device
        )

        y_dw_2d = causal_conv1d_fn(
            pre_dw_2d,
            conv_weights,
            conv_bias,
            conv_state,
            query_start_loc,
            cache_indices=page_indices,
            has_initial_state=has_initial_state,
            activation=None,
        )  # [D, T]

        y_bn = self.bn(y_dw_2d.unsqueeze(0))      # [1, D, T]
        y_act = F.silu(y_bn)
        y_out = self.pw2(y_act).squeeze(0).transpose(0, 1)  # [T, D]
        return y_out

    def forward(self, x: torch.Tensor):
        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if isinstance(attn_meta_all, dict) and x.dim() == 2:
            return self.forward_cuda(x)
        return self.forward_ref(x)

class ConformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads: int,
        k_conv: int,
        ff_mult: int,
        attn_window: int,
        cache_config: CacheConfig,
        dtype: torch.dtype,
        prefix: str,
    ):
        super().__init__()
        self.ln_ff1 = nn.LayerNorm(d_model)
        self.ff1 = ConformerFFN(d_model, ff_mult)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = RelPosSelfAttention(d_model, n_heads, attn_window, cache_config, prefix=f"{prefix}.attn")
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, k_conv, prefix=f"{prefix}.conv", cache_config=cache_config, dtype=dtype)
        self.ln_ff2 = nn.LayerNorm(d_model)
        self.ff2 = ConformerFFN(d_model, ff_mult)
        self.ln_out = nn.LayerNorm(d_model)
        self.fc_factor = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_ff1(x)
        y = self.ff1(y)
        x = x + y * self.fc_factor

        y = self.ln_attn(x)
        y = self.attn(y)
        x = x + y

        y = self.ln_conv(x)
        y = self.conv(y)
        x = x + y

        y = self.ln_ff2(x)
        y = self.ff2(y)
        x = x + y * self.fc_factor

        x = self.ln_out(x)
        return x


class FastConformerCTC(nn.Module):
    """FastConformerCTC for vLLM."""
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: FastConformerCTCConfig = vllm_config.model_config.hf_config
        self.config = config

        self.d_model = config.d_model
        self.dtype = vllm_config.model_config.dtype

        self.vocab_size = config.ctc.get("vocab_size")
        assert self.vocab_size is not None, "config missing vocab_size"
        self.blank_id = config.blank_id

        subs = config.subsampling or {}
        assert subs.get("type", "dw_striding") == "dw_striding", "Only 'dw_striding' subsampling is implemented"
        assert subs.get("factor", 8) == 8, "Only 8x subsampling is assumed"
        assert subs.get("channels", 256) == 256, "Only 256 channels are supported"
        mid_ch = subs.get("channels", 256)

        # TODO: see comment above the NemoSubsample8x2D class
        # self.subsample = NemoSubsample8x2D(d_out=self.d_model, mid_ch=mid_ch, mels=80)
        from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
        self.subsample = ConvSubsampling(
            subsampling="dw_striding",
            subsampling_factor=8,
            feat_in=80,
            feat_out=self.d_model,
            conv_channels=mid_ch,
            is_causal=True,
        ).to(self.dtype).eval()

        att_window = int(config.att_left_ctx + config.att_right_ctx)
        assert att_window > 0, "att_window must be positive"

        self.prefix = prefix

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                n_heads=config.num_attention_heads,
                k_conv=config.k_conv,
                ff_mult=config.ff_mult,
                attn_window=att_window,
                cache_config=vllm_config.cache_config,
                dtype=self.dtype,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.n_layers)
        ])

        self.proj = nn.Linear(self.d_model, self.vocab_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise Exception("not applicable for this model")

    def _remove_ragged_format(self, x: torch.Tensor, pre_subsample: bool = True) -> torch.Tensor:
        if _is_dummy_run():
            return x.unsqueeze(0)
        attn_metadata = get_forward_context().attn_metadata
        ctx: FastConformerMetadata = list(attn_metadata.values())[0]
        num_seqs = ctx.num_reqs
        seq_lens = ctx.query_start_loc[1:] - ctx.query_start_loc[:-1]
        if not torch.all(seq_lens == seq_lens[0]):
            raise NotImplementedError(
                "Ragged batch processing for variable sequence lengths is not supported"
            )
        batch_size = num_seqs
        time_dim = seq_lens[0].item()
        feature_dim = x.shape[-1]
        if pre_subsample:
            time_dim = time_dim * 8
        return x.view(batch_size, time_dim, feature_dim)

    def _add_ragged_format(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)

    def _forward_attn_only(self, x: torch.Tensor) -> torch.Tensor:
        # x = self._remove_ragged_format(x, pre_subsample=False)
        x = self.blocks[0].attn(x)
        # x = self._add_ragged_format(x)
        return x
    
    def _forward_conv_only(self, x: torch.Tensor) -> torch.Tensor:
        # x = self._remove_ragged_format(x, pre_subsample=False)
        x = self.blocks[0].conv(x)
        # x = self._add_ragged_format(x)
        return x

    def forward(
        self,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,            # unused
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert inputs_embeds is not None, "inputs_embeds must be provided as [T, F]"
        x = inputs_embeds
        assert x.dim() == 2, f"expected [T, F], got shape {tuple(x.shape)}"

        # used in tests
        if self.config.attn_only:
            return self._forward_attn_only(x)
        if self.config.conv_only:
            return self._forward_conv_only(x)

        T, F = x.shape
        assert F == 640, f"expected feature dim=80*8, got {F}"
        x = x.view(T, 8, 80).reshape(T * 8, 80)

        x = self._remove_ragged_format(x)

        # _check("/home/scratch.jdaw_coreai/landrew/tmp/audio_signal_pre.pt", x)

        length = x.new_full(
                (x.size(0),), x.size(1), dtype=torch.int64, device=x.device
            )

        x, _ = self.subsample(x, length)
        # _check("/home/scratch.jdaw_coreai/landrew/tmp/audio_signal.pt", x)
        xscale = math.sqrt(self.d_model)
        x = (x * xscale)
        # _check("/home/scratch.jdaw_coreai/landrew/tmp/audio_signal_post.pt", x)

        # x = x[:,1:,:]

        for i, blk in enumerate(self.blocks):
            x = blk(x)                 # [T/8, D]
            # _check(f"/home/scratch.jdaw_coreai/landrew/tmp/audio_signal_post_{i}.pt", x)
        # x = self.attn(x)

        # slice last n frames
        x = x[:,1:,:]
        x = self._add_ragged_format(x)
        return x

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # hidden_states: [T, D]
        return self.proj(hidden_states)  # [T, vocab]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        nemo = {name: tensor for name, tensor in weights}

        loaded_pairs: list[tuple[str, str]] = []
        skipped: list[tuple[str, str]] = []

        model_params = dict(self.named_parameters())
        loaded_param_names: set[str] = set()

        # helper to copy tensors with shape checking
        def copy_(dst_param: torch.nn.Parameter, src: torch.Tensor,
                dst_name: str, src_name: str):
            if dst_param.shape != src.shape:
                if src.dim() == 3 and src.shape[-1] == 1 and dst_param.shape == src.shape[:2]:
                    dst_param.data.copy_(src.squeeze(-1))
                else:
                    skipped.append((src_name,
                                    f"shape {tuple(src.shape)} -> {dst_name} {tuple(dst_param.shape)}"))
                    return
            else:
                dst_param.data.copy_(src)
            loaded_pairs.append((src_name, dst_name))
            loaded_param_names.add(dst_name)

        # sub_map = [
        #     ("encoder.pre_encode.conv.0.weight", self.subsample.conv0.weight, "subsample.conv0.weight"),
        #     ("encoder.pre_encode.conv.0.bias",   self.subsample.conv0.bias,   "subsample.conv0.bias"),
        #     ("encoder.pre_encode.conv.2.weight", self.subsample.conv2.weight, "subsample.conv2.weight"),
        #     ("encoder.pre_encode.conv.2.bias",   self.subsample.conv2.bias,   "subsample.conv2.bias"),
        #     ("encoder.pre_encode.conv.3.weight", self.subsample.conv3.weight, "subsample.conv3.weight"),
        #     ("encoder.pre_encode.conv.3.bias",   self.subsample.conv3.bias,   "subsample.conv3.bias"),
        #     ("encoder.pre_encode.conv.5.weight", self.subsample.conv5.weight, "subsample.conv5.weight"),
        #     ("encoder.pre_encode.conv.5.bias",   self.subsample.conv5.bias,   "subsample.conv5.bias"),
        #     ("encoder.pre_encode.conv.6.weight", self.subsample.conv6.weight, "subsample.conv6.weight"),
        #     ("encoder.pre_encode.conv.6.bias",   self.subsample.conv6.bias,   "subsample.conv6.bias"),
        #     ("encoder.pre_encode.out.weight",    self.subsample.out.weight,   "subsample.out.weight"),
        #     ("encoder.pre_encode.out.bias",      self.subsample.out.bias,     "subsample.out.bias"),
        # ]
        # inside load_weights(), after nemo = {...}
        # print([k for k in nemo if "pre_encode" in k][:50])
        # print([k for k in nemo if "ctc" in k or "decoder" in k][:50])
        sub_map = [
            ("encoder.pre_encode.conv.0.weight", self.subsample.conv[0].weight, "subsample.conv.0.weight"),
            ("encoder.pre_encode.conv.0.bias",   self.subsample.conv[0].bias,   "subsample.conv.0.bias"),
            ("encoder.pre_encode.conv.2.weight", self.subsample.conv[2].weight, "subsample.conv.2.weight"),
            ("encoder.pre_encode.conv.2.bias",   self.subsample.conv[2].bias,   "subsample.conv.2.bias"),
            ("encoder.pre_encode.conv.3.weight", self.subsample.conv[3].weight, "subsample.conv.3.weight"),
            ("encoder.pre_encode.conv.3.bias",   self.subsample.conv[3].bias,   "subsample.conv.3.bias"),
            ("encoder.pre_encode.conv.5.weight", self.subsample.conv[5].weight, "subsample.conv.5.weight"),
            ("encoder.pre_encode.conv.5.bias",   self.subsample.conv[5].bias,   "subsample.conv.5.bias"),
            ("encoder.pre_encode.conv.6.weight", self.subsample.conv[6].weight, "subsample.conv.6.weight"),
            ("encoder.pre_encode.conv.6.bias",   self.subsample.conv[6].bias,   "subsample.conv.6.bias"),
            ("encoder.pre_encode.out.weight",    self.subsample.out.weight,   "subsample.out.weight"),
            ("encoder.pre_encode.out.bias",      self.subsample.out.bias,     "subsample.out.bias"),
        ]
        for n_src, p_dst, n_dst in sub_map:
            if n_src in nemo:
                copy_(p_dst, nemo[n_src], n_dst, n_src)

        for i, blk in enumerate(self.blocks):
            base = f"encoder.layers.{i}"

            ln_pairs = [
                (f"{base}.norm_feed_forward1.weight", blk.ln_ff1.weight, f"blocks.{i}.ln_ff1.weight"),
                (f"{base}.norm_feed_forward1.bias",   blk.ln_ff1.bias,   f"blocks.{i}.ln_ff1.bias"),
                (f"{base}.norm_self_att.weight",      blk.ln_attn.weight,f"blocks.{i}.ln_attn.weight"),
                (f"{base}.norm_self_att.bias",        blk.ln_attn.bias,  f"blocks.{i}.ln_attn.bias"),
                (f"{base}.norm_conv.weight",          blk.ln_conv.weight,f"blocks.{i}.ln_conv.weight"),
                (f"{base}.norm_conv.bias",            blk.ln_conv.bias,  f"blocks.{i}.ln_conv.bias"),
                (f"{base}.norm_feed_forward2.weight", blk.ln_ff2.weight, f"blocks.{i}.ln_ff2.weight"),
                (f"{base}.norm_feed_forward2.bias",   blk.ln_ff2.bias,   f"blocks.{i}.ln_ff2.bias"),
                (f"{base}.norm_out.weight",           blk.ln_out.weight, f"blocks.{i}.ln_out.weight"),
                (f"{base}.norm_out.bias",             blk.ln_out.bias,   f"blocks.{i}.ln_out.bias"),
            ]
            for n_src, p_dst, n_dst in ln_pairs:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            ffn1 = [
                (f"{base}.feed_forward1.linear1.weight", blk.ff1.linear1.weight, f"blocks.{i}.ff1.linear1.weight"),
                (f"{base}.feed_forward1.linear1.bias",   blk.ff1.linear1.bias,   f"blocks.{i}.ff1.linear1.bias"),
                (f"{base}.feed_forward1.linear2.weight", blk.ff1.linear2.weight, f"blocks.{i}.ff1.linear2.weight"),
                (f"{base}.feed_forward1.linear2.bias",   blk.ff1.linear2.bias,   f"blocks.{i}.ff1.linear2.bias"),
            ]
            for n_src, p_dst, n_dst in ffn1:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            attn = [
                (f"{base}.self_attn.linear_q.weight", blk.attn.q_proj.weight, f"blocks.{i}.attn.q_proj.weight"),
                (f"{base}.self_attn.linear_q.bias",   blk.attn.q_proj.bias,   f"blocks.{i}.attn.q_proj.bias"),
                (f"{base}.self_attn.linear_k.weight", blk.attn.k_proj.weight, f"blocks.{i}.attn.k_proj.weight"),
                (f"{base}.self_attn.linear_k.bias",   blk.attn.k_proj.bias,   f"blocks.{i}.attn.k_proj.bias"),
                (f"{base}.self_attn.linear_v.weight", blk.attn.v_proj.weight, f"blocks.{i}.attn.v_proj.weight"),
                (f"{base}.self_attn.linear_v.bias",   blk.attn.v_proj.bias,   f"blocks.{i}.attn.v_proj.bias"),
                (f"{base}.self_attn.linear_out.weight", blk.attn.o_proj.weight, f"blocks.{i}.attn.o_proj.weight"),
                (f"{base}.self_attn.linear_out.bias",   blk.attn.o_proj.bias,   f"blocks.{i}.attn.o_proj.bias"),
                (f"{base}.self_attn.linear_pos.weight", blk.attn.linear_pos.weight, f"blocks.{i}.attn.linear_pos.weight"),
                (f"{base}.self_attn.pos_bias_u",        blk.attn.pos_bias_u,        f"blocks.{i}.attn.pos_bias_u"),
                (f"{base}.self_attn.pos_bias_v",        blk.attn.pos_bias_v,        f"blocks.{i}.attn.pos_bias_v"),
            ]
            for n_src, p_dst, n_dst in attn:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            conv = [
                (f"{base}.conv.pointwise_conv1.weight", blk.conv.pw1.weight, f"blocks.{i}.conv.pw1.weight"),
                (f"{base}.conv.pointwise_conv1.bias",   blk.conv.pw1.bias,   f"blocks.{i}.conv.pw1.bias"),
                (f"{base}.conv.depthwise_conv.weight",  blk.conv.dw.weight,  f"blocks.{i}.conv.dw.weight"),
                (f"{base}.conv.depthwise_conv.bias",    blk.conv.dw.bias,    f"blocks.{i}.conv.dw.bias"),
                (f"{base}.conv.batch_norm.weight",      blk.conv.bn.weight,  f"blocks.{i}.conv.bn.weight"),
                (f"{base}.conv.batch_norm.bias",        blk.conv.bn.bias,    f"blocks.{i}.conv.bn.bias"),
                (f"{base}.conv.batch_norm.running_mean", blk.conv.bn.running_mean, f"blocks.{i}.conv.bn.running_mean"),
                (f"{base}.conv.batch_norm.running_var",  blk.conv.bn.running_var,  f"blocks.{i}.conv.bn.running_var"),
                (f"{base}.conv.batch_norm.num_batches_tracked", blk.conv.bn.num_batches_tracked, f"blocks.{i}.conv.bn.num_batches_tracked"),
                (f"{base}.conv.pointwise_conv2.weight", blk.conv.pw2.weight, f"blocks.{i}.conv.pw2.weight"),
                (f"{base}.conv.pointwise_conv2.bias",   blk.conv.pw2.bias,   f"blocks.{i}.conv.pw2.bias"),
            ]
            for n_src, p_dst, n_dst in conv:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            ffn2 = [
                (f"{base}.feed_forward2.linear1.weight", blk.ff2.linear1.weight, f"blocks.{i}.ff2.linear1.weight"),
                (f"{base}.feed_forward2.linear1.bias",   blk.ff2.linear1.bias,   f"blocks.{i}.ff2.linear1.bias"),
                (f"{base}.feed_forward2.linear2.weight", blk.ff2.linear2.weight, f"blocks.{i}.ff2.linear2.weight"),
                (f"{base}.feed_forward2.linear2.bias",   blk.ff2.linear2.bias,   f"blocks.{i}.ff2.linear2.bias"),
            ]
            for n_src, p_dst, n_dst in ffn2:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            with torch.no_grad():
                attn_mod = blk.attn
                w_cat = torch.cat([
                    attn_mod.q_proj.weight,
                    attn_mod.k_proj.weight,
                    attn_mod.v_proj.weight,
                ], dim=0)
                b_cat = torch.cat([
                    attn_mod.q_proj.bias,
                    attn_mod.k_proj.bias,
                    attn_mod.v_proj.bias,
                ], dim=0)
                attn_mod.qkv_weight.copy_(w_cat)
                attn_mod.qkv_bias.copy_(b_cat)
                attn_mod._qkv_fused_ready = True

        head_w = "ctc_decoder.decoder_layers.0.weight"
        head_b = "ctc_decoder.decoder_layers.0.bias"
        if head_w in nemo:
            copy_(self.proj.weight, nemo[head_w], "proj.weight", head_w)
        if head_b in nemo:
            copy_(self.proj.bias, nemo[head_b], "proj.bias", head_b)

        loaded_src = {src for (src, _) in loaded_pairs}

        print(f"[load_weights] Loaded {len(loaded_pairs)} tensors.")

        if skipped:
            print(f"[load_weights] Skipped {len(skipped)} tensors (showing first 40):")
            for n, why in skipped[:40]:
                if n not in loaded_src:
                    print(f"  - {n}: {why}")

        unused_model_params = sorted(set(model_params.keys()) - loaded_param_names)
        if unused_model_params:
            print(f"[load_weights] Model params with NO checkpoint match ({len(unused_model_params)} shown first 40):")
            for n in unused_model_params[:40]:
                print(f"  - {n} : shape {tuple(model_params[n].shape)}")

        if not skipped and not unused_model_params:
            print("[load_weights] all weights loaded successfully.")
