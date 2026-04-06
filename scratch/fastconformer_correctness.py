import os
import json
import time
import argparse

import torch
import numpy as np

from safetensors.torch import save_file

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _nemo_imports():
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    from omegaconf import OmegaConf
    return nemo_asr, ConformerEncoder, OmegaConf


def _seed_all(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)



def _make_random_mel_packets(steps: int, mels: int = 80, seed: int = 0):
    _seed_all(seed)
    # Each step provides 8 frames collapsed to a single embedding row of size 80*8
    D_IN = mels * 8
    return [torch.randn(1, D_IN).to(torch.float32) for _ in range(steps)]


async def run_vllm_hidden_states(bundle_dir: str, packets: list[torch.Tensor]):
    req_id = "fc-correctness"

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import SamplingParams
    from vllm.inputs.data import EmbedsPrompt

    engine_args = AsyncEngineArgs(
        model=bundle_dir,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        block_size=128,
        enable_prompt_embeds=True,
        enforce_eager=True,
        return_hidden_states=True,
        skip_tokenizer_init=True,
        dtype="float32",
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    # concateate packets into a single tensor
    print(f"packets[0].shape: {packets[0].shape}")
    packets_tensor = torch.cat(packets, dim=0)
    print(f"packets_tensor.shape: {packets_tensor.shape}")
    packets = [packets_tensor]

    gen_iter = engine.generate(
        request_id=req_id,
        prompt=EmbedsPrompt(prompt_embeds=packets[0]),
        sampling_params=SamplingParams(max_tokens=len(packets)),
        is_streaming=True,
    )

    hs_all = []
    try:
        out = await gen_iter.__anext__()
        hs = out.outputs[0].hidden_states[-1]
        print(f"hs.shape: {hs.shape}")
        hs_all.append(hs.detach().cpu())
    except StopAsyncIteration:
        pass

    for i in range(1, len(packets)):
        await engine.append_request(request_id=req_id, input_embeds=packets[i])
        try:
            out = await gen_iter.__anext__()
            hs = out.outputs[0].hidden_states[-1]
            hs_all.append(hs.detach().cpu())
        except StopAsyncIteration:
            break

    return hs_all


def _stack_mels_from_packets(packets: list[torch.Tensor], mels: int = 80) -> torch.Tensor:
    # Packets were [1, 80*8] per step; reconstruct [B=1, T=steps*8, F=80]
    rows = [p.reshape(1, 8, mels) for p in packets]
    x = torch.cat(rows, dim=1)  # [1, steps*8, 80]
    return x


def _nemo_forward(conformer: torch.nn.Module, mel_frames: torch.Tensor) -> torch.Tensor:
    # Input mel_frames: [B, T, F] where F=80
    # NeMo expects audio_signal shape [B, F, T] when bypass_pre_encode=False
    B, T, F = mel_frames.shape
    audio_signal = mel_frames.permute(0, 2, 1).contiguous()  # [B, F, T]
    print(f"[debugging] audio_signal.shape {audio_signal.shape} audio_signal.dtype {audio_signal.dtype}")
    length = torch.full((B,), T, dtype=torch.long, device=mel_frames.device)
    with torch.no_grad():
        encoded, encoded_len = conformer(
            audio_signal=audio_signal,
            length=length,
            bypass_pre_encode=False,
        )
    return encoded[:, :, 1:]


def _print_diff_metrics(vllm_hs: torch.Tensor, nemo_hs: torch.Tensor, label: str):
    a = vllm_hs.detach().float().cpu()
    b = nemo_hs.detach().float().cpu()
    if a.shape != b.shape:
        print(f"[{label}] SHAPE MISMATCH: vLLM {tuple(a.shape)} vs NeMo {tuple(b.shape)}")
        return
    diff = (a - b).abs()
    rel = diff / (b.abs() + 1e-6)
    print(f"[{label}] shape={tuple(a.shape)} | mean abs diff={diff.mean():.6f} | max abs diff={diff.max():.6f} | mean rel diff={rel.mean():.6f}")


def build_nemo_conformer(d_model, n_layers, n_heads, ff_mult, k_conv, att_left_ctx, att_right_ctx):
    nemo_asr, ConformerEncoder, OmegaConf = _nemo_imports()
    # Minimal config consistent with vLLM FastConformerCTC
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
        # since we are benchmarking a streaming encoder
        conv_context_size="causal",
        causal_downsampling=True,
        # att_context_size=[att_left_ctx, att_right_ctx],
        att_context_size=[71,0]
    )
    enc = enc.to(torch.float32)
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


def _save_packets(out_dir: str, packets: list[torch.Tensor]):
    arr = torch.stack([p.squeeze(0).to(torch.float32).cpu() for p in packets], dim=0)  # [steps, 80*8]
    torch.save(arr, os.path.join(out_dir, "packets.pt"))


def _load_packets(out_dir: str) -> list[torch.Tensor]:
    arr = torch.load(os.path.join(out_dir, "packets.pt"), map_location="cpu")  # [steps, 80*8]
    return [arr[i:i+1, :] for i in range(arr.shape[0])]


def _save_hidden_states(out_dir: str, hs: torch.Tensor):
    torch.save(hs.to(torch.float32).cpu(), os.path.join(out_dir, "nemo_hidden_states.pt"))


def _load_hidden_states(out_dir: str) -> torch.Tensor:
    return torch.load(os.path.join(out_dir, "nemo_hidden_states.pt"), map_location="cpu")


def run_nemo_mode(out_dir: str, steps: int, seed: int):
    _seed_all(seed)

    nemo_enc, cfg = build_nemo_conformer(
        d_model=512, n_layers=17, n_heads=8, ff_mult=4, k_conv=9, att_left_ctx=70, att_right_ctx=1
    )

    os.makedirs(out_dir, exist_ok=True)

    nemo_sd = nemo_enc.state_dict()
    remapped = {f"encoder.{k}": v.to(torch.float32).contiguous() for k, v in nemo_sd.items()}
    save_file(remapped, os.path.join(out_dir, "model.safetensors"))

    fc_cfg = {
        "architectures": ["FastConformerCTC"],
        "model_type": "fastconformer_ctc",
        "hidden_size": 80 * 8,
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "ff_mult": cfg["ff_mult"],
        "k_conv": cfg["k_conv"],
        "subsampling": {"type": "dw_striding", "factor": 8, "channels": 256, "total_stride": 8},
        "att_left_ctx": cfg["att_left_ctx"],
        "att_right_ctx": cfg["att_right_ctx"],
        "ctc": {"vocab_size": 1024},
        "blank_id": 0,
        "vocab_size": 1024,
        "frontend": {"sample_rate": None, "n_fft": None, "n_mels": 80},
        "tokenizer": {"type": "sentencepiece", "path": "sentencepiece.bpe.model"},
        "notes": "NeMo ConformerEncoder weights for vLLM correctness test"
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(fc_cfg, f, indent=2)

    packets = _make_random_mel_packets(steps=steps, seed=seed)
    print(f"len(packets): {len(packets)}")
    print(f"packets[0].shape: {packets[0].shape}")
    mel = _stack_mels_from_packets(packets)
    nemo_y = _nemo_forward(nemo_enc, mel).squeeze(0).transpose(-1,-2) # [steps, D]
    print(f"nemo_y.shape: {nemo_y.shape}")
    assert nemo_y.shape == (steps, cfg["d_model"])

    _save_packets(out_dir, packets)
    _save_hidden_states(out_dir, nemo_y)

    meta = {"steps": steps, "seed": seed, "d_model": cfg["d_model"], "n_layers": cfg["n_layers"], "n_heads": cfg["n_heads"]}
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] NeMo mode complete. Bundle + data saved to: {out_dir}")


async def run_vllm_mode(out_dir: str):
    packets = _load_packets(out_dir)
    nemo_y = _load_hidden_states(out_dir)  # [steps, D]

    vllm_hs_list = await run_vllm_hidden_states(out_dir, packets)
    vllm_y = torch.stack([t.squeeze(0) for t in vllm_hs_list], dim=0)

    _print_diff_metrics(vllm_y, nemo_y, label="EncoderOut")
    if vllm_y.shape == nemo_y.shape:
        for idx in range(vllm_y.shape[0]):
            _print_diff_metrics(vllm_y[idx], nemo_y[idx], label=f"Frame {idx}")


async def main():
    parser = argparse.ArgumentParser(description="fastconformer correctness test")
    parser.add_argument("--mode", choices=["nemo", "vllm"], required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = "/home/scratch.jdaw_coreai/landrew/fastconformer_correctness_tests/"
    os.makedirs(outdir, exist_ok=True)

    if args.mode == "nemo":
        run_nemo_mode(outdir, steps=args.steps, seed=args.seed)
    else:
        await run_vllm_mode(outdir)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

