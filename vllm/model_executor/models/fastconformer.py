# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

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
    """
    Exact NeMo-style 2D subsampler for FastConformer:
      - Input (no batch): [T, F] with F=mels (typically 80)
      - Output: [T/8, D] with D=d_out (512)
      - Matches checkpoint keys:
        encoder.pre_encode.conv.{0,2,3,5,6}.* and encoder.pre_encode.out.*
    """
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
        # x: [T, F]
        T, F = x.shape
        assert F == self.mels, f"Expected mel dim {self.mels}, got {F}"
        y = x.view(1, 1, T, F)             # [B=1, C=1, T, F]
        y = self.act(self.conv0(y))        # [1, 256, T/2, F/2]
        y = self.act(self.conv2(y))        # [1, 256, T/4, F/4]
        y = self.act(self.conv3(y))        # [1, 256, T/4, F/4]
        y = self.act(self.conv5(y))        # [1, 256, T/8, ~11]
        y = self.act(self.conv6(y))        # [1, 256, T/8, 11]
        B, C, T8, Fp = y.shape
        # Safety check: must be 11 so that C*Fp == 2816 for the Linear layer
        if C * Fp != 2816:
            raise RuntimeError(f"Subsampler produced C*F'={C}*{Fp}={C*Fp}, expected 2816.")
        y = y.permute(2, 0, 1, 3).contiguous().view(T8, C * Fp)  # [T/8, 256*11]
        y = self.out(y)                 # [T/8, d_out]
        return y


class ConformerFFN(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, pdrop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc2(F.silu(self.fc1(y)))
        return x + self.drop(y) * scale


class RelPosSelfAttention(nn.Module):
    """scores = (q + u) @ k^T + (q + v) @ r^T."""
    def __init__(self, d_model: int, num_heads: int, window: int):
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
        # x: [H, T, 2T-1] -> [H, T, T] (Transformer-XL trick)
        H, T, _ = x.shape
        x = F.pad(x, (1, 0))                      # [H, T, 2T]
        x = x.view(H, -1, T)                      # [H, (T+ (T-1)), T] == [H, 2T-1 + 1 - 1, T] but using the known layout
        x = x[:, 1:]                               # drop first row to shift
        return x[:, :T]                            # [H, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape
        H, Dh = self.h, self.dh

        q = self.q_proj(x).view(T, H, Dh).permute(1, 0, 2)  # [H, T, Dh]
        k = self.k_proj(x).view(T, H, Dh).permute(1, 0, 2)  # [H, T, Dh]
        v = self.v_proj(x).view(T, H, Dh).permute(1, 0, 2)  # [H, T, Dh]

        # (q + u) @ k^T
        q_with_u = q + self.pos_bias_u.unsqueeze(1)         # [H, T, Dh]
        content_scores = torch.matmul(q_with_u, k.transpose(-2, -1))  # [H, T, T]

        rel = self._build_rel_sin_table(T, D, x.device, x.dtype)      # [2T-1, D]
        rel = self.linear_pos(rel)                                     # [2T-1, D]
        rel = rel.view(2 * T - 1, H, Dh).permute(1, 0, 2).contiguous() # [H, 2T-1, Dh]

        q_with_v = q + self.pos_bias_v.unsqueeze(1)                    # [H, T, Dh]
        rel_scores = torch.matmul(q_with_v, rel.transpose(-2, -1))     # [H, T, 2T-1]
        rel_scores = self._rel_shift(rel_scores)                       # [H, T, T]

        scores = (content_scores + rel_scores) * (Dh ** -0.5)          # [H, T, T]

        mask = build_local_band_mask(T, self.window, device=x.device, dtype=scores.dtype)  # [T, T]
        scores = scores + mask  # broadcast over H

        attn = F.softmax(scores, dim=-1)                               # [H, T, T]
        y = torch.matmul(attn, v)                                      # [H, T, Dh]
        y = y.permute(1, 0, 2).contiguous().view(T, D)                 # [T, D]
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
        # x: [T, D]
        y = self.ln(x)                      # [T, D]
        y = y.transpose(0, 1).unsqueeze(0)  # [1, D, T]  (N, C, L)

        y = self.pw1(y)                     # [1, 2D, T]
        a, b = y.chunk(2, dim=1)            # split along channel dim
        y = a * torch.sigmoid(b)            # [1, D, T]

        y = self.dw(y)                      # [1, D, T]
        y = self.bn(y)                      # [1, D, T]  BN now sees C=D (good)
        y = F.silu(y)

        y = self.pw2(y)                     # [1, D, T]
        y = y.squeeze(0).transpose(0, 1)    # [T, D]

        return x + y


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_conv: int, ff_mult: int, attn_window: int, pdrop: float = 0.0):
        super().__init__()
        self.ff1 = ConformerFFN(d_model, ff_mult, pdrop)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = RelPosSelfAttention(d_model, n_heads, attn_window)
        self.drop = nn.Dropout(pdrop)
        self.conv = ConformerConvModule(d_model, k_conv)
        self.ff2 = ConformerFFN(d_model, ff_mult, pdrop)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x, scale=0.5)
        y = self.ln_attn(x); y = self.attn(y)
        x = x + self.drop(y)
        x = self.conv(x)
        x = self.ff2(x, scale=0.5)
        x = self.ln_out(x)
        return x


class FastConformerCTC(nn.Module):
    """FastConformerCTC for vLLM, batchless I/O."""
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
        mid_ch = subs.get("channels", 256)
        self.subsample = NemoSubsample8x2D(d_out=self.d_model, mels=80)

        att_window = int(config.att_left_ctx + config.att_right_ctx)
        assert att_window > 0, "att_window must be positive"

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                n_heads=config.n_heads,
                k_conv=config.k_conv,
                ff_mult=config.ff_mult,
                attn_window=att_window,
                pdrop=0.0,
            )
            for _ in range(config.n_layers)
        ])

        self.proj = nn.Linear(self.d_model, self.vocab_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise Exception("not applicable for this model")

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

        x = self.subsample(x)

        for blk in self.blocks:
            x = blk(x)                 # [T/8, D]

        return x

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # hidden_states: [T, D]
        return self.proj(hidden_states)  # [T, vocab]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        pass
