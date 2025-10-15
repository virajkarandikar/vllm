# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.sequence import IntermediateTensors

from vllm.v1.attention.backends.fastconformer_attn import (
    FastConformerBackend,
    FastConformerMetadata,
)
from vllm.forward_context import get_forward_context
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, FastConformerSpec

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
    def __init__(self, d_model: int, ff_mult: int = 4, pdrop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        # x: [B, T, D]
        y = self.ln(x)
        y = self.fc2(F.silu(self.fc1(y)))
        return x + self.drop(y) * scale


class FastConformerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        sliding_window: int,
        num_kv_heads: int,
        head_dim: int,
        prefix: str,
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.dtype = torch.bfloat16
        self.sliding_window = sliding_window
        self.num_kv_heads = num_kv_heads
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerSpec(
            block_size=self.sliding_window, # `block_size` controls the shape of the KV cache
            sliding_window=self.sliding_window,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> AttentionBackend:
        return FastConformerBackend


class RelPosSelfAttention(nn.Module):
    """scores = (q + u) @ k^T + (q + v) @ r^T."""
    def __init__(self, d_model: int, num_heads: int, window: int, prefix: str):
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
        self.cache_prefix = f"{self.prefix}.kv_cache"

        self.cache = FastConformerCache(
            sliding_window=self.window,
            num_kv_heads=self.h,
            head_dim=self.dh,
            prefix=self.cache_prefix,
        )

    @staticmethod
    def _build_rel_sin_table(T: int, D: int, device, dtype):
        # Standard sinusoidal table for relative positions [-T+1, ..., 0, ..., +T-1]
        # shape: [2T-1, D]
        pos = torch.arange(-(T - 1), T, device=device, dtype=dtype).unsqueeze(1)   # [2T-1, 1]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) * (-math.log(10000.0) / D))  # [D/2]
        sin = torch.sin(pos * div)  # [2T-1, D/2]
        cos = torch.cos(pos * div)  # [2T-1, D/2]
        table = torch.zeros((2 * T - 1, D), device=device, dtype=dtype)
        table[:, 0::2] = sin
        table[:, 1::2] = cos
        return table  # [2T-1, D]

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, 2T-1] -> [B, H, T, T] (Transformer-XL trick, batched)
        B, H, T, m = x.shape
        x = F.pad(x, (1, 0))  # pad width in dim=3 (2T-1 -> 2T)
        x = x.view(B, H, -1, T)  # [B, H, (T + (T-1)), T]
        x = x[:, :, 1:, :]  # drop the first element in new "S"-dim
        return x[:, :, :T, :]  # return [B, H, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # NOTE: attention metadata is not populated for dummy runs
            return self.forward_no_cache(x)
        else:
            return self.forward_cache(x, attn_metadata[self.cache_prefix])

    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        return self._attention_forward(q, k, v)

    def forward_cache(self, x: torch.Tensor, ctx: FastConformerMetadata) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k_new = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        v_new = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]

        with torch.no_grad():
            kv_cache = self.cache.kv_cache[0]  # (2, n_pages, window, H, Dh)
            k_cache = kv_cache[0][ctx.slot_mapping, ...]  # [B, window, H, Dh]
            v_cache = kv_cache[1][ctx.slot_mapping, ...]  # [B, window, H, Dh]

        k_cache = k_cache.permute(0, 2, 1, 3).contiguous()  # [B, H, window, Dh]
        v_cache = v_cache.permute(0, 2, 1, 3).contiguous()  # [B, H, window, Dh]

        k_cat = torch.cat([k_cache, k_new], dim=2)  # [B, H, window+T, Dh]
        v_cat = torch.cat([v_cache, v_new], dim=2)  # [B, H, window+T, Dh]

        # update kv cache by rolling left T
        k_cache_new = torch.cat(
            [k_cache[:, :, T:, :].permute(0, 2, 1, 3), k_new.permute(0, 2, 1, 3)], dim=1
        )  # [B, window, H, Dh]
        v_cache_new = torch.cat(
            [v_cache[:, :, T:, :].permute(0, 2, 1, 3), v_new.permute(0, 2, 1, 3)], dim=1
        )  # [B, window, H, Dh]

        with torch.no_grad():
            self.cache.kv_cache[0][0][ctx.slot_mapping, ...] = k_cache_new
            self.cache.kv_cache[0][1][ctx.slot_mapping, ...] = v_cache_new

        return self._attention_forward(q, k_cat, v_cat)

    def _attention_forward(self, q, k, v) -> torch.Tensor:
        # q, k, v: [B, H, T/S, Dh]
        B, H, T, Dh = q.shape
        S = k.shape[2]
        D = H * Dh
        device = q.device
        dtype = q.dtype

        # (q + u) @ k^T
        pos_bias_u = self.pos_bias_u.unsqueeze(0).unsqueeze(2)  # [1, H, 1, Dh]
        q_with_u = q + pos_bias_u                               # [B, H, T, Dh]
        content_scores = torch.matmul(q_with_u, k.transpose(-2, -1))  # [B, H, T, S]

        # TODO: can we cache this?
        rel = self._build_rel_sin_table(S, D, device, dtype)            # [2S-1, D]
        rel = self.linear_pos(rel)                                      # [2S-1, D]
        rel = rel.view(2 * S - 1, H, Dh).permute(1, 0, 2).contiguous()  # [H, 2S-1, Dh]

        # Compute (q + v) @ r^T with correct batching
        pos_bias_v = self.pos_bias_v.unsqueeze(0).unsqueeze(2)          # [1, H, 1, Dh]
        q_with_v = q + pos_bias_v                                      # [B, H, T, Dh]
        rel_t = rel.transpose(1, 2)                                    # [H, Dh, 2S-1]

        # Compute: [B, H, T, Dh] x [H, Dh, 2S-1] -> [B, H, T, 2S-1]
        rel_scores = torch.einsum('bhtd,hdm->bhtm', q_with_v, rel_t)   # [B, H, T, 2S-1]
        rel_scores = self._rel_shift(rel_scores)                       # [B, H, T, S]

        scores = (content_scores + rel_scores) * (Dh ** -0.5)          # [B, H, T, S]

        mask = build_local_band_mask(T, S, device=device, dtype=scores.dtype)  # [T, S]
        scores = scores + mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,S], broadcast over B, H

        attn = F.softmax(scores, dim=-1)                               # [B, H, T, S]

        y = torch.matmul(attn, v)                                      # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)               # [B, T, D]
        return self.o_proj(y)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, k: int = 9):
        super().__init__()
        assert k % 2 == 1
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw  = nn.Conv1d(d_model, d_model, k, padding=(k-1)//2,
                             groups=d_model)
        self.bn  = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        y = self.ln(x)                         # [B, T, D]
        y = y.transpose(1, 2)                  # [B, D, T] (N, C, L)

        y = self.pw1(y)                        # [B, 2D, T]
        a, b = y.chunk(2, dim=1)               # split along channel dim
        y = a * torch.sigmoid(b)               # [B, D, T]

        y = self.dw(y)                         # [B, D, T]
        y = self.bn(y)                         # [B, D, T]
        y = F.silu(y)                          # [B, D, T]

        y = self.pw2(y)                        # [B, D, T]
        y = y.transpose(1, 2)                  # [B, T, D]

        return x + y                           # [B, T, D]


class ConformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads: int,
        k_conv: int,
        ff_mult: int,
        attn_window: int,
        prefix: str,
    ):
        super().__init__()
        self.ff1 = ConformerFFN(d_model, ff_mult)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = RelPosSelfAttention(d_model, n_heads, attn_window, prefix=f"{prefix}.attn")
        self.conv = ConformerConvModule(d_model, k_conv)
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
                n_heads=config.n_heads,
                k_conv=config.k_conv,
                ff_mult=config.ff_mult,
                attn_window=att_window,
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
        ctx: FastConformerMetadata = attn_metadata.values()[0]
        num_seqs = ctx.num_reqs
        seq_lens = ctx.query_start_loc[1:] - ctx.query_start_loc[:-1]
        if not torch.all(seq_lens == seq_lens[0]):
            raise NotImplementedError(
                "Ragged batch processing for variable sequence lengths is not supported"
            )
        batch_size = num_seqs
        time_dim = seq_lens[0].item()
        feature_dim = x.shape[-1]
        return x.view(batch_size, time_dim, feature_dim)


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
