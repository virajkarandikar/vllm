import os
import json
import time
import argparse

import torch
import numpy as np

from safetensors.torch import save_file

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = "cuda"

def _nemo_imports():
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    from omegaconf import OmegaConf
    return nemo_asr, ConformerEncoder, OmegaConf

def _seed_all(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def _make_random_frames(batch_size: int, num_frames: int, feat_dim: int, seed: int = 0):
    _seed_all(seed)
    # [batch_size, num_frames, feat_dim]
    return torch.randn(batch_size, num_frames, feat_dim, device=DEVICE).to(torch.float32)

def _nemo_forward(conformer: torch.nn.Module, packets: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    # packets: [batch, feats, seq]
    audio_signal = packets.contiguous()
    with torch.no_grad():
        encoded, encoded_len = conformer(
            audio_signal=audio_signal,
            length=length,
            bypass_pre_encode=False,
        )
    return encoded

def build_nemo_conformer(
    d_model, n_layers, n_heads, ff_mult, k_conv, att_left_ctx, att_right_ctx
):
    nemo_asr, ConformerEncoder, OmegaConf = _nemo_imports()
    cfg = OmegaConf.create({
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ff_expansion_factor": ff_mult,
        "conv_kernel_size": k_conv,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "att_context_size": [att_left_ctx, att_right_ctx],
    })
    enc = ConformerEncoder(
        feat_in=80,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ff_expansion_factor=ff_mult,
        conv_kernel_size=k_conv,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_channels=256,
        conv_context_size="causal",
        causal_downsampling=True,
        att_context_size=[att_left_ctx, att_right_ctx]
    )
    enc = enc.to(torch.float32).to(DEVICE)
    enc.eval()
    return enc, {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ff_mult": ff_mult,
        "k_conv": k_conv,
        "att_left_ctx": att_left_ctx,
        "att_right_ctx": att_right_ctx,
    }

def benchmark_conformer(
    conformer: torch.nn.Module,
    all_frames: torch.Tensor,
    context_size: int = 71,
    warmup: int = 10,
    bench_frames: int = 100,
    progress: bool = True,
    mode: str = "eager",
):
    results = []
    total_time = 0.0
    batch_size, total_frames, feat_dim = all_frames.shape
    assert total_frames >= context_size + bench_frames

    B = batch_size
    T = context_size + 1

    for i in range(warmup):
        i_start = i
        i_end = i + context_size + 1
        window = all_frames[:, i_start : i_end, :].transpose(1,2)  # [B, feats, T]
        inp = window.to(DEVICE)
        length = torch.full((B,), T, dtype=torch.long, device=DEVICE)
        _ = _nemo_forward(conformer, inp, length)

    times = []
    if mode == "eager":
        for fidx in range(bench_frames):
            i_start = fidx
            i_end = fidx + context_size + 1
            window = all_frames[:, i_start : i_end, :].transpose(1,2)  # [B, feats, T]
            inp = window.to(DEVICE)
            length = torch.full((B,), T, dtype=torch.long, device=DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _nemo_forward(conformer, inp, length)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)
            if progress and (fidx % 10 == 0 or fidx == bench_frames-1):
                print(f"Frame {fidx+1}/{bench_frames}, batch size: {batch_size}, latency: {elapsed*1000.0:.3f} ms")
    elif mode == "graph":
        static_inp = torch.empty((B, feat_dim, T), dtype=all_frames.dtype, device=DEVICE)
        static_length = torch.full((B,), T, dtype=torch.long, device=DEVICE)

        first_window = all_frames[:, 0 : context_size + 1, :].transpose(1,2)  # [B, feats, T]
        static_inp.copy_(first_window)
        torch.cuda.synchronize()
        warm_out = _nemo_forward(conformer, static_inp, static_length)
        static_out = torch.empty_like(warm_out)

        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            tmp = _nemo_forward(conformer, static_inp, static_length)
            static_out.copy_(tmp)
        torch.cuda.synchronize()

        for fidx in range(bench_frames):
            i_start = fidx
            i_end = fidx + context_size + 1
            window = all_frames[:, i_start : i_end, :].transpose(1,2)  # [B, feats, T]
            inp = window.to(DEVICE)
            static_inp.copy_(inp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)
            if progress and (fidx % 10 == 0 or fidx == bench_frames-1):
                print(f"Frame {fidx+1}/{bench_frames}, batch size: {batch_size}, latency: {elapsed*1000.0:.3f} ms")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total = sum(times)
    per_frame = total / bench_frames
    print(f"Total time for {bench_frames} frames: {total*1000.0:.3f} ms")
    print(f"Inter-token latency: {per_frame*1000.0:.3f} ms")
    return total, per_frame, times

async def main():
    parser = argparse.ArgumentParser(description="fastconformer benchmarking script")
    parser.add_argument("--frames", type=int, default=100, help="number of frames to generate/benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup iterations")
    parser.add_argument("--feats", type=int, default=80, help="hidden size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--ctx", type=int, default=71, help="context frames for attention")
    parser.add_argument("--outdir", type=str, default="/home/scratch.jdaw_coreai/landrew/fastconformer_correctness_tests/")
    parser.add_argument("--progress", action="store_true", help="show progress")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for benchmarking")
    parser.add_argument("--mode", type=str, choices=["eager", "graph"], default="eager", help="benchmark mode: eager or CUDA graph")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    n_layers = 17
    n_heads = 8
    ff_mult = 4
    k_conv = 9
    att_left_ctx = args.ctx
    att_right_ctx = 0

    nemo_enc, cfg = build_nemo_conformer(
        d_model=args.feats, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult,
        k_conv=k_conv, att_left_ctx=att_left_ctx, att_right_ctx=att_right_ctx
    )
    nemo_enc = nemo_enc.to(torch.float32).to(DEVICE)
    nemo_sd = nemo_enc.state_dict()
    remapped = {f"encoder.{k}": v.to(torch.float32).contiguous().to("cpu") for k, v in nemo_sd.items()}
    save_file(remapped, os.path.join(outdir, "model.safetensors"))

    num_needed_frames = args.ctx + args.frames + args.warmup
    all_frames = _make_random_frames(
        batch_size=args.batch_size,
        num_frames=num_needed_frames,
        feat_dim=args.feats,
        seed=args.seed
    ).to(DEVICE)

    print(f"running benchmark: {args.frames} frames, {args.ctx}-frame context, {args.warmup} warmup, batch size: {args.batch_size}, mode: {args.mode}")
    total, per_frame, times = benchmark_conformer(
        nemo_enc, all_frames, context_size=args.ctx, warmup=args.warmup,
        bench_frames=args.frames, progress=args.progress, mode=args.mode
    )
    print(f"=== Benchmark results ===")
    print(f"   Total frames: {args.frames}")
    print(f"   Context size: {args.ctx}")
    print(f"   Feature dim: {args.feats}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Mode: {args.mode}")
    print(f"   Inter-token latency: {per_frame*1000.0:.3f} ms")
    print(f"   Mean: {np.mean(times)*1000.0:.3f} ms, Std: {np.std(times)*1000.0:.3f} ms")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

