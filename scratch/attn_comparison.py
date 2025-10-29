#!/usr/bin/env python3
import math
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

try:
    from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
except ImportError:
    torch_sdpa = None

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_determinism(seed: int = 0):
    torch.manual_seed(seed)
    # If you see cublas non-determinism errors, export one of:
    #   CUBLAS_WORKSPACE_CONFIG=:16:8   or   :4096:8
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def stats(name: str, a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    ma = d.max().item()
    me = d.mean().item()
    print(f"{name:10s} | max_abs={ma:.6g}  mean_abs={me:.6g}")
    return d

def worst_entries(diff: torch.Tensor, topk=5):
    flat = diff.flatten()
    vals, idxs = torch.topk(flat, k=min(topk, flat.numel()))
    coords = []
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        coords.append((v, idx))
    return coords

# ---------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------

@dataclass
class Config:
    d_model: int = 64
    num_heads: int = 4
    window: int = 8  # inclusive in reference path (<= W)
    dtype: str = "float32"
    device: str = "cpu"

class RelPosSelfAttentionStandalone(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.h = cfg.num_heads
        self.dh = cfg.d_model // cfg.num_heads
        self.window = int(cfg.window)

        D = cfg.d_model
        self.q_proj = nn.Linear(D, D, bias=True)
        self.k_proj = nn.Linear(D, D, bias=True)
        self.v_proj = nn.Linear(D, D, bias=True)
        self.o_proj = nn.Linear(D, D, bias=True)

        self.linear_pos = nn.Linear(D, D, bias=False)
        # u and v biases (Transformer-XL)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.dh))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.dh))

        # init similar to standard Transformer layers
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.linear_pos.weight)

    # ---------- Reference (Transformer-XL style) ----------
    def forward_ref(self, q_inp: torch.Tensor, kv_inp: torch.Tensor) -> torch.Tensor:
        # q_inp: [B, Q=1, D]   kv_inp: [B, K=71, D]
        B, Q, D = q_inp.shape
        _, K, Dk = kv_inp.shape
        H, Dh = self.h, self.dh
        assert D == H * Dh and Dk == D

        # Projections: [B,H,Q,Dh], [B,H,K,Dh], [B,H,K,Dh]
        q = self.q_proj(q_inp).view(B, Q, H, Dh).transpose(1, 2).contiguous()         # [B,H,1,Dh]
        k = self.k_proj(kv_inp).view(B, K, H, Dh).transpose(1, 2).contiguous()        # [B,H,71,Dh]
        v = self.v_proj(kv_inp).view(B, K, H, Dh).transpose(1, 2).contiguous()        # [B,H,71,Dh]

        # Sinusoidal positions for lengths Q and K
        device, idtype = q_inp.device, q_inp.dtype
        pos_idx = torch.arange(K - 1, -K, -1, device=device)[:K]  # [K]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / D))
        sin = torch.sin(pos_idx[:, None].to(torch.float32) * div[None, :])  # [K,D/2]
        cos = torch.cos(pos_idx[:, None].to(torch.float32) * div[None, :])
        pos = torch.zeros(K, D, device=device, dtype=torch.float32)
        pos[:, 0::2] = sin
        pos[:, 1::2] = cos

        # linear_pos and reshape for [B,H,K,Dh]
        p = self.linear_pos(pos.to(idtype)).view(K, H, Dh).permute(1, 0, 2).contiguous()
        p = p.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,K,Dh]

        # biases
        q_with_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2) # [B,H,1,Dh]
        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2) # [B,H,1,Dh]

        # scores: content term
        scores_ac = torch.matmul(q_with_u, k.transpose(-2, -1))   # [B,H,1,71]

        # positional term
        raw_bd = torch.matmul(q_with_v, p.transpose(-2, -1))      # [B,H,1,71]
        scores_bd = raw_bd

        # combine + scale
        scores = (scores_ac + scores_bd) * (Dh ** -0.5)

        # (Optional masking: for causal/relative window, could adjust if desired)
        W = int(self.window)
        if W > 0 and W < K:
            q_idx = torch.zeros(Q, device=device, dtype=torch.long)         # 0 for query
            k_idx = torch.arange(K, device=device, dtype=torch.long)        # 0...K-1 for keys
            allowed = (k_idx <= q_idx[:, None]) & ((q_idx[:, None] - k_idx) <= W)
            shape_mask = allowed.unsqueeze(0).unsqueeze(0)   # [1,1,1,K]
            scores = scores.masked_fill(~shape_mask, float("-inf"))

        # stable softmax (fp32)
        scores_fp32 = scores.to(torch.float32)
        scores_fp32 = scores_fp32 - torch.amax(scores_fp32, dim=-1, keepdim=True)
        probs = torch.softmax(scores_fp32, dim=-1).to(idtype)

        # attend V
        ctx = torch.matmul(probs, v)  # [B,H,1,Dh]
        out = ctx.transpose(1, 2).contiguous().view(B, Q, D)
        return self.o_proj(out)

    def _build_shaw_band_bias(self, q: torch.Tensor, W: int, K: int) -> torch.Tensor:
        """Build Shaw-style band bias: shape [B,H,Q,2W+1], already scaled by Dh**-0.5."""
        B, H, Q, Dh = q.shape
        D = H * Dh
        device, dtype = q.device, q.dtype

        # For each possible relative position from -W to W
        deltas = torch.arange(-W, W + 1, device=device)[:, None].to(torch.float32)  # [2W+1,1]
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / D))
        sin = torch.sin(deltas * div)  # [2W+1, D/2]
        cos = torch.cos(deltas * div)
        rel = torch.zeros((2 * W + 1, D), device=device, dtype=torch.float32)
        rel[:, 0::2] = sin
        rel[:, 1::2] = cos

        rel = self.linear_pos(rel.to(dtype))                       # [2W+1, D]
        rel = rel.view(2 * W + 1, H, Dh).permute(1, 2, 0)          # [H, Dh, 2W+1]

        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)   # [B,H,Q,Dh]
        band_bias = torch.einsum("bhtd,hdm->bhtm", q_with_v, rel)  # [B,H,Q,2W+1]
        band_bias = band_bias * (Dh ** -0.5)                       # match QK scale
        return band_bias

    def forward_flex(self, q_inp: torch.Tensor, kv_inp: torch.Tensor) -> torch.Tensor:
        """
        Implements the same math as forward_ref using torch.nn.attention.flex_attention,
        but now with Q=1 (query), K=71 (keys).
        """
        B, Q, D = q_inp.shape
        _, K, Dk = kv_inp.shape
        H, Dh = self.h, self.dh
        assert D == H * Dh and Dk == D

        # Project to [B,H,Q, Dh] and [B,H,K, Dh]
        q = self.q_proj(q_inp).view(B, Q, H, Dh).transpose(1, 2).contiguous()  # [B,H,1,Dh]
        k = self.k_proj(kv_inp).view(B, K, H, Dh).transpose(1, 2).contiguous() # [B,H,71,Dh]
        v = self.v_proj(kv_inp).view(B, K, H, Dh).transpose(1, 2).contiguous() # [B,H,71,Dh]

        W = int(self.window)
        band_bias = self._build_shaw_band_bias(q, W, K)  # [B,H,1,2W+1]

        def score_mod(score, b, h, q_idx, kv_idx):
            # rel = q_idx - kv_idx
            rel = q_idx - kv_idx
            rel = torch.clamp(rel, -W, W) + W # index in 2W+1 range
            return score + band_bias[b.long(), h.long(), q_idx.long(), rel.long()]

        def mask_mod(b, h, q_idx, kv_idx):
            # Windowed and causal: keys only <= query, within W
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) <= W)

        # Build block_mask for (Q=1, K=71)
        block_mask = create_block_mask(
            mask_mod,
            None,  # per-batch static mask
            None,  # per-head static mask
            Q,     # num query positions (1)
            K,     # num kv positions (71)
            device=q_inp.device,
            BLOCK_SIZE=(128, 128),
        )

        scale = Dh ** -0.5
        print(f"[scratch_debug] q.shape: {q.shape}")
        print(f"[scratch_debug] k.shape: {k.shape}")
        print(f"[scratch_debug] v.shape: {v.shape}")

        ctx = flex_attention(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": 128, "BLOCK_N": 128},
        )  # [B,H,1,Dh]

        out = ctx.transpose(1, 2).contiguous().view(B, Q, D)
        return self.o_proj(out)

    def forward_sdpa(self, q_inp: torch.Tensor, kv_inp: torch.Tensor) -> torch.Tensor:
        """
        Forward kernel using PyTorch's scaled_dot_product_attention for benchmarking.
        Uses the *Shaw-style* band bias as the additive attn_mask so that outputs match
        the numerics of the Shaw/relative-position reference.
        """
        if torch_sdpa is None:
            raise RuntimeError("torch.nn.functional.scaled_dot_product_attention not available in this PyTorch.")
        B, Q, D = q_inp.shape
        _, K, Dk = kv_inp.shape
        H, Dh = self.h, self.dh
        assert D == H * Dh and Dk == D

        # [B, Q, D] -> [B, Q, H, Dh] -> [B, H, Q, Dh]
        q = self.q_proj(q_inp).view(B, Q, H, Dh).transpose(1, 2)  # [B, H, Q, Dh]
        k = self.k_proj(kv_inp).view(B, K, H, Dh).transpose(1, 2) # [B, H, K, Dh]
        v = self.v_proj(kv_inp).view(B, K, H, Dh).transpose(1, 2) # [B, H, K, Dh]

        # Merge heads for SDPA: [B, H, Q, Dh] -> [B*H, Q, Dh]
        q_sdpa = q.reshape(B * H, Q, Dh)
        k_sdpa = k.reshape(B * H, K, Dh)
        v_sdpa = v.reshape(B * H, K, Dh)

        # Shaw-style band bias mask [B, H, Q, K]
        W = int(self.window)
        # Build Shaw-style band bias [B, H, Q, 2W+1]
        band_bias = self._build_shaw_band_bias(q, W, K) # [B, H, Q, 2W+1]
        # Place it into attn_mask [B, H, Q, K] by mapping the relative position
        # For each (q_idx, k_idx), rel_pos = q_idx - k_idx
        attn_mask = torch.zeros(B, H, Q, K, device=q.device, dtype=band_bias.dtype)
        # fill Shaw band bias in the correct relative position (rel = q_idx - k_idx)
        for q_idx in range(Q):
            for k_idx in range(K):
                rel = q_idx - k_idx
                if -W <= rel <= W:
                    rel_idx = rel + W
                    attn_mask[:, :, q_idx, k_idx] = band_bias[:, :, q_idx, rel_idx]
                else:
                    attn_mask[:, :, q_idx, k_idx] = 0.0  # no bias for distant keys

        # Now set -inf for disallowed window/casual positions (match other paths)
        if W > 0 and W < K:
            for q_idx in range(Q):
                for k_idx in range(K):
                    allowed = (k_idx <= q_idx) and ((q_idx - k_idx) <= W)
                    if not allowed:
                        attn_mask[:, :, q_idx, k_idx] = float('-inf')

        # Reshape attn_mask to [B*H, Q, K]
        attn_mask = attn_mask.reshape(B * H, Q, K)

        t0 = time.perf_counter()
        y = torch_sdpa(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,  # [B*H, Q, K]
            dropout_p=0.0,
            is_causal=False,
        )
        t1 = time.perf_counter()
        print(f"[scratch_debug] forward_sdpa time: {(t1 - t0) * 1000} ms")
        # [B*H, Q, Dh] -> [B, H, Q, Dh]
        y = y.view(B, H, Q, Dh)

        # Restore [B, Q, D]
        y = y.transpose(1, 2).contiguous().view(B, Q, D)
        return self.o_proj(y)

# ---------------------------------------------------------------------
# Main: build module, run, compare
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Q", type=int, default=1)
    parser.add_argument("--K", type=int, default=71)
    parser.add_argument("--D", type=int, default=64*8)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--W", type=int, default=8, help="inclusive window (<= W)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_determinism(args.seed)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    act_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    cfg = Config(
        d_model=args.D,
        num_heads=args.H,
        window=args.W,
        dtype=args.dtype,
        device=args.device,
    )

    mod = RelPosSelfAttentionStandalone(cfg).to(device)
    mod = mod.to(dtype=act_dtype)
    mod.eval()

    # Random input for query and key/value (same dtype as module)
    q = torch.randn(args.B, args.Q, args.D, device=device, dtype=act_dtype)
    kv = torch.randn(args.B, args.K, args.D, device=device, dtype=act_dtype)
    print(f"[scratch_debug] q.shape: {q.shape}   kv.shape: {kv.shape}")

    with torch.inference_mode():
        # Reference
        t0 = time.perf_counter()
        y_ref  = mod.forward_ref(q, kv)
        t1 = time.perf_counter()
        print(f"[scratch_debug] forward_ref time: {(t1 - t0) * 1000} ms")

        # Flex attention (3 dummy warmups)
        for _ in range(3):
            dummyq = torch.randn(args.B, args.Q, args.D, device=device, dtype=act_dtype)
            dummykv = torch.randn(args.B, args.K, args.D, device=device, dtype=act_dtype)
            _ = mod.forward_flex(dummyq, dummykv)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_flex = mod.forward_flex(q, kv)
        t1 = time.perf_counter()
        print(f"[scratch_debug] forward_flex time: {(t1 - t0) * 1000} ms")

        # (NEW) PyTorch SDPA/FlashAttention-like benchmarking
        if torch_sdpa is not None:
            for _ in range(3):
                dummyq = torch.randn(args.B, args.Q, args.D, device=device, dtype=act_dtype)
                dummykv = torch.randn(args.B, args.K, args.D, device=device, dtype=act_dtype)
                _ = mod.forward_sdpa(dummyq, dummykv)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            y_sdpa = mod.forward_sdpa(q, kv)
            t1 = time.perf_counter()
            print(f"[scratch_debug] forward_sdpa time: {(t1 - t0) * 1000} ms")
        else:
            print("torch.nn.functional.scaled_dot_product_attention not available, skipping SDPA benchmark.")

    print(f"Shapes: y_ref={tuple(y_ref.shape)}, y_flex={tuple(y_flex.shape)}, y_sdpa={tuple(y_sdpa.shape)}")
    d = stats("OUTPUT", y_flex, y_ref)
    _d = stats("OUTPUT", y_sdpa, y_ref)

    # Show a few worst positions (flattened index)
    tops = worst_entries(d, topk=8)
    if len(tops):
        print("Top diffs (abs, flat_idx):")
        for v, idx in tops:
            print(f"  {v:.6g} @ {idx}")

    # Optional per-element relative error (guard zeros)
    eps = 1e-12
    rel = (y_flex - y_ref).abs() / (y_ref.abs() + eps)
    print(f"RelErr     | max={rel.max().item():.6g}  mean={rel.mean().item():.6g}")

if __name__ == "__main__":
    main()
