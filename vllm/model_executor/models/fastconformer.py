# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec, FastConformerConvSpec

from vllm.transformers_utils.configs.fastconformer import FastConformerCTCConfig
import math

def build_local_band_mask(T: int, window: int, device, dtype) -> torch.Tensor:
    mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    idx = torch.arange(T, device=device)
    k = idx.view(1, T)
    q = idx.view(T, 1)
    allowed = (k <= q) & (k >= (q - (window - 1)))
    mask = mask.masked_fill(allowed, 0.0)
    return mask


class NemoSubsample8x2D(nn.Module):
    def __init__(self, d_out: int, mels: int = 80):
        super().__init__()
        self.mels = mels
        self.conv0 = nn.Conv2d(1, 256, kernel_size=3, stride=(2, 2), padding=(1, 1))            # conv.0
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=(2, 2), padding=(1, 1), groups=256)  # conv.2 (DW)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)                     # conv.3 (PW)
        # Asymmetric pad on freq to make F_out == 11 for F_in==80 (so 256*11 = 2816 for Linear)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=(2, 2), padding=(1, 2), groups=256)  # conv.5 (DW)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)                     # conv.6 (PW)
        self.act = nn.SiLU()
        self.out = nn.Linear(256 * 11, d_out)  # 2816 -> 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape
        assert F == self.mels, f"Expected mel dim {self.mels}, got {F}"
        y = x.view(B, 1, T, F)             # [B, C=1, T, F]
        y = self.act(self.conv0(y))        # [B, 256, T/2, F/2]
        y = self.act(self.conv2(y))        # [B, 256, T/4, F/4]
        y = self.act(self.conv3(y))        # [B, 256, T/4, F/4]
        y = self.act(self.conv5(y))        # [B, 256, T/8, ~11]
        y = self.act(self.conv6(y))        # [B, 256, T/8, 11]
        B, C, T8, Fp = y.shape
        # Safety check: must be 11 so that C*Fp == 2816 for the Linear layer
        if C * Fp != 2816:
            raise RuntimeError(f"Subsampler produced C*F'={C}*{Fp}={C*Fp}, expected 2816.")
        y = y.permute(2, 0, 1, 3).contiguous().view(B, T8, C * Fp)  # [B, T/8, 256*11]
        y = self.out(y)                 # [B, T/8, d_out]
        return y


class ConformerFFN(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)

    def forward(self, x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        # x: [B, T, D]
        y = self.ln(x)
        y = self.fc2(F.silu(self.fc1(y)))
        return x + y * scale

class FastConformerConvCache(torch.nn.Module, AttentionLayerBase):
    def __init__(self, d_model: int, k: int, prefix: str, cache_config: CacheConfig):
        super().__init__()
        self.dtype = torch.bfloat16
        self.d_model = int(d_model)
        self.k = int(k)
        self.left_ctx = self.k - 1  # L
        # TODO: this is a hack to get the page size to match attn page size
        # we should try to get smth more durable to work
        self.left_shape = self.left_ctx * 32
        self.prefix = prefix
        self.cache_config = cache_config
        self.kv_cache = [torch.tensor([])]

        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            # TODO: i think this should just be block_size=1?
            # need to re-check how the attn metadata is constructed
            block_size=self.cache_config.block_size,
            # shape=(self.left_ctx, self.d_model),
            shape=(self.left_shape, self.d_model),
            dtype=self.dtype,
        )

    def forward(self):
        pass

    def get_attn_backend(self) -> AttentionBackend:
        return FastConformerBackend



class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window: int, cache_config: CacheConfig, prefix: str):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.window = int(window)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.dh))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.dh))

        self.prefix = prefix

        self.attn = Attention(
            num_heads=self.h,
            head_size=self.dh,
            scale=self.dh**-0.5,
            num_kv_heads=self.h,
            cache_config=CacheConfig(
                sliding_window=self.window,
                cache_dtype="auto",
                block_size=cache_config.block_size,
                calculate_kv_scales=False,
            ),
            prefix=self.prefix,
            attn_backend=FlexAttentionBackend
        )

        # TODO: verify whether this is needed
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        if cache_config.block_size != 128:
            # attn page size must match conv page size
            raise Exception(f"attn cache block size must be 128, got {cache_config.block_size}")

    def _build_shaw_band_bias(self, q: torch.Tensor, W: int) -> torch.Tensor:
        B, H, T, Dh = q.shape
        D = H * Dh
        device, dtype = q.device, q.dtype

        deltas = torch.arange(-(W - 1), W, device=device)
        pos = deltas[:, None].to(dtype)  # [2W-1, 1]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype)
                        * (-math.log(10000.0) / D))
        sin = torch.sin(pos * div)  # [2W-1, D/2]
        cos = torch.cos(pos * div)
        rel = torch.zeros((2 * W - 1, D), device=device, dtype=dtype)
        rel[:, 0::2] = sin
        rel[:, 1::2] = cos
        rel = self.linear_pos(rel)                              # [2W-1, D]
        rel = rel.view(2 * W - 1, H, Dh).permute(1, 2, 0)       # [H, Dh, 2W-1]

        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)  # [B,H,T,Dh]
        band_bias = torch.einsum("bhtd,hdm->bhtm", q_with_v, rel) # [B,H,T,2W-1]
        return band_bias

    def _make_shaw_score_mod(self, band_bias: torch.Tensor, W: int):
        # band_bias: [B, H, T, 2W-1]
        @torch.jit.script_if_tracing
        def score_mod(score: torch.Tensor,
                    b: torch.Tensor, h: torch.Tensor,
                    q_idx: torch.Tensor, kv_idx: torch.Tensor,
                    physical_q: torch.Tensor = None) -> torch.Tensor:
            rel = q_idx - kv_idx
            low = -(W - 1); high = (W - 1)
            rel = torch.clamp(rel, low, high)
            idx = rel + (W - 1)  # [0..2W-2]
            return score + band_bias[b.long(), h.long(), q_idx.long(), idx.long()]
        return score_mod

    def _prepare_flex(self, attn_metadata: FlexAttentionMetadata, score_mod):
        # TODO: verify whether this is needed. for example,
        # - is block_mask already set?
        # - is score_mod already set?
        # - is transformed_score_mod already set?
        attn_metadata.sliding_window = int(self.window)

        attn_metadata.score_mod = score_mod
        attn_metadata.transformed_score_mod = attn_metadata.get_transformed_score_mod()

        attn_metadata.block_mask = attn_metadata._build_block_mask_direct()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()  # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()  # [B,H,T,Dh]
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()  # [B,H,T,Dh]

        q_plus_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)            # [B,H,T,Dh]

        band_bias = self._build_shaw_band_bias(q, self.window)
        score_mod = self._make_shaw_score_mod(band_bias, self.window)

        query = q_plus_u.reshape(-1, H, Dh)
        key   = k.reshape(-1, H, Dh)
        value = v.reshape(-1, H, Dh)

        fctx = get_forward_context()
        attn_metadata_all = fctx.attn_metadata
        if isinstance(attn_metadata_all, dict):
            attn_metadata: FlexAttentionMetadata = attn_metadata_all[self.prefix]
            self._prepare_flex(attn_metadata, score_mod=score_mod)
        else:
            # dummy run; compute normal attention
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (Dh ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, value)
            out = attn_output.view(B, T, H, Dh).transpose(1, 2).contiguous().view(B, T, D)
            return self.o_proj(out)

        self_kv_cache = self.attn.kv_cache[fctx.virtual_engine]

        out_buf = torch.empty_like(query)
        out = self.attn.impl.forward(
            layer=self,
            query=query,
            key=key,
            value=value,
            kv_cache=self_kv_cache,
            attn_metadata=attn_metadata,
            output=out_buf,
        )
        out = out.view(B, T, H, Dh).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, k: int, prefix: str, cache_config: CacheConfig):
        super().__init__()
        assert k % 2 == 1
        self.prefix = prefix
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw  = nn.Conv1d(d_model, d_model, k, padding=(k-1)//2,
                             groups=d_model)
        self.bn  = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)

        self.conv_cache = FastConformerConvCache(
            d_model=d_model,
            k=k,
            prefix=f"{prefix}.conv_cache",
            cache_config=cache_config,
        )

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        assert D == self.d_model

        y = self.ln(x)              # [B,T,D]
        y = y.transpose(1, 2)       # [B,D,T] (N,C,L)
        y = self.pw1(y)             # [B,2D,T]
        a, b = y.chunk(2, dim=1)
        pre_dw = a * torch.sigmoid(b)   # [B,D,T]

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if isinstance(attn_meta_all, dict):
            attn_metadata: FastConformerMetadata = attn_meta_all[self.conv_cache.prefix]
        else:
            # dummy run; compute normal conv
            y_dw = self.dw(pre_dw)
            y_dw = self.bn(y_dw)
            y_dw = F.silu(y_dw)
            y = self.pw2(y_dw)
            y = y.transpose(1, 2)
            return x + y

        block_table = attn_metadata.block_table_tensor    # [B, n_blocks]
        page_indices = block_table[:, 0]

        store = self.conv_cache.kv_cache[fctx.virtual_engine]    # [num_pages, L, D]

        hist = store[page_indices]              # [B, L, D]
        hist = hist.transpose(1, 2).contiguous()   # [B, D, L]

        x_cat = torch.cat([hist, pre_dw], dim=-1)   # [B, D, L+T]

        y_dw = F.conv1d(
            x_cat, self.dw.weight, self.dw.bias,
            stride=1, padding=0, dilation=1, groups=D
        )[:,:,-T:]                                        # [B,D,T]
        y_dw = self.bn(y_dw)
        y_dw = F.silu(y_dw)

        y = self.pw2(y_dw)                       # [B,D,T]
        y = y.transpose(1, 2)                    # [B,T,D]

        # update cache
        new_hist = x_cat[:, :, -self.conv_cache.left_shape:]  # [B,D,L]
        store[page_indices] = new_hist.transpose(1, 2)  # back to [B,L,D]

        return x + y


class ConformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads: int,
        k_conv: int,
        ff_mult: int,
        attn_window: int,
        cache_config: CacheConfig,
        prefix: str,
    ):
        super().__init__()
        self.ff1 = ConformerFFN(d_model, ff_mult)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = RelPosSelfAttention(d_model, n_heads, attn_window, cache_config, prefix=f"{prefix}.attn")
        self.conv = ConformerConvModule(d_model, k_conv, prefix=f"{prefix}.conv", cache_config=cache_config)
        self.ff2 = ConformerFFN(d_model, ff_mult)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x, scale=0.5)
        y = self.ln_attn(x)
        y = self.attn(y)
        x = x + y
        x = self.conv(x)
        x = self.ff2(x, scale=0.5)
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

        self.vocab_size = config.ctc.get("vocab_size")
        assert self.vocab_size is not None, "config missing vocab_size"
        self.blank_id = config.blank_id

        subs = config.subsampling or {}
        assert subs.get("type", "dw_striding") == "dw_striding", "Only 'dw_striding' subsampling is implemented"
        assert subs.get("factor", 8) == 8, "Only 8x subsampling is assumed"
        mid_ch = subs.get("channels", 256) # unused for now
        self.subsample = NemoSubsample8x2D(d_out=self.d_model, mels=80)

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
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.n_layers)
        ])

        self.proj = nn.Linear(self.d_model, self.vocab_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise Exception("not applicable for this model")

    def _remove_ragged_format(self, x: torch.Tensor) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # attn_metadata is None during dummy runs
            return x.unsqueeze(0)
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
        return x.view(batch_size, time_dim * 8, feature_dim)


    def _add_ragged_format(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)

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

        T, F = x.shape
        assert F == 640, f"expected feature dim=80*8, got {F}"
        x = x.view(T, 8, 80).reshape(T * 8, 80)

        x = self._remove_ragged_format(x)

        x = self.subsample(x)

        for blk in self.blocks:
            x = blk(x)                 # [T/8, D]

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

        sub_map = [
            ("encoder.pre_encode.conv.0.weight", self.subsample.conv0.weight, "subsample.conv0.weight"),
            ("encoder.pre_encode.conv.0.bias",   self.subsample.conv0.bias,   "subsample.conv0.bias"),
            ("encoder.pre_encode.conv.2.weight", self.subsample.conv2.weight, "subsample.conv2.weight"),
            ("encoder.pre_encode.conv.2.bias",   self.subsample.conv2.bias,   "subsample.conv2.bias"),
            ("encoder.pre_encode.conv.3.weight", self.subsample.conv3.weight, "subsample.conv3.weight"),
            ("encoder.pre_encode.conv.3.bias",   self.subsample.conv3.bias,   "subsample.conv3.bias"),
            ("encoder.pre_encode.conv.5.weight", self.subsample.conv5.weight, "subsample.conv5.weight"),
            ("encoder.pre_encode.conv.5.bias",   self.subsample.conv5.bias,   "subsample.conv5.bias"),
            ("encoder.pre_encode.conv.6.weight", self.subsample.conv6.weight, "subsample.conv6.weight"),
            ("encoder.pre_encode.conv.6.bias",   self.subsample.conv6.bias,   "subsample.conv6.bias"),
            ("encoder.pre_encode.out.weight",    self.subsample.out.weight,   "subsample.out.weight"),
            ("encoder.pre_encode.out.bias",      self.subsample.out.bias,     "subsample.out.bias"),
        ]
        for n_src, p_dst, n_dst in sub_map:
            if n_src in nemo:
                copy_(p_dst, nemo[n_src], n_dst, n_src)

        for i, blk in enumerate(self.blocks):
            base = f"encoder.layers.{i}"

            ln_pairs = [
                (f"{base}.norm_feed_forward1.weight", blk.ff1.ln.weight, f"blocks.{i}.ff1.ln.weight"),
                (f"{base}.norm_feed_forward1.bias",   blk.ff1.ln.bias,   f"blocks.{i}.ff1.ln.bias"),
                (f"{base}.norm_self_att.weight",      blk.ln_attn.weight,f"blocks.{i}.ln_attn.weight"),
                (f"{base}.norm_self_att.bias",        blk.ln_attn.bias,  f"blocks.{i}.ln_attn.bias"),
                (f"{base}.norm_conv.weight",          blk.conv.ln.weight,f"blocks.{i}.conv.ln.weight"),
                (f"{base}.norm_conv.bias",            blk.conv.ln.bias,  f"blocks.{i}.conv.ln.bias"),
                (f"{base}.norm_feed_forward2.weight", blk.ff2.ln.weight, f"blocks.{i}.ff2.ln.weight"),
                (f"{base}.norm_feed_forward2.bias",   blk.ff2.ln.bias,   f"blocks.{i}.ff2.ln.bias"),
                (f"{base}.norm_out.weight",           blk.ln_out.weight, f"blocks.{i}.ln_out.weight"),
                (f"{base}.norm_out.bias",             blk.ln_out.bias,   f"blocks.{i}.ln_out.bias"),
            ]
            for n_src, p_dst, n_dst in ln_pairs:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            ffn1 = [
                (f"{base}.feed_forward1.linear1.weight", blk.ff1.fc1.weight, f"blocks.{i}.ff1.fc1.weight"),
                (f"{base}.feed_forward1.linear1.bias",   blk.ff1.fc1.bias,   f"blocks.{i}.ff1.fc1.bias"),
                (f"{base}.feed_forward1.linear2.weight", blk.ff1.fc2.weight, f"blocks.{i}.ff1.fc2.weight"),
                (f"{base}.feed_forward1.linear2.bias",   blk.ff1.fc2.bias,   f"blocks.{i}.ff1.fc2.bias"),
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
                (f"{base}.conv.pointwise_conv2.weight", blk.conv.pw2.weight, f"blocks.{i}.conv.pw2.weight"),
                (f"{base}.conv.pointwise_conv2.bias",   blk.conv.pw2.bias,   f"blocks.{i}.conv.pw2.bias"),
            ]
            for n_src, p_dst, n_dst in conv:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            ffn2 = [
                (f"{base}.feed_forward2.linear1.weight", blk.ff2.fc1.weight, f"blocks.{i}.ff2.fc1.weight"),
                (f"{base}.feed_forward2.linear1.bias",   blk.ff2.fc1.bias,   f"blocks.{i}.ff2.fc1.bias"),
                (f"{base}.feed_forward2.linear2.weight", blk.ff2.fc2.weight, f"blocks.{i}.ff2.fc2.weight"),
                (f"{base}.feed_forward2.linear2.bias",   blk.ff2.fc2.bias,   f"blocks.{i}.ff2.fc2.bias"),
            ]
            for n_src, p_dst, n_dst in ffn2:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

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
