import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

def _get_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")



def _build_nemo_from_config(model_dir: str, dtype: torch.dtype):
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = json.load(open(cfg_path, "r"))

    d_model = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    n_heads = int(cfg["n_heads"])
    ff_mult = int(cfg["ff_mult"])
    k_conv = int(cfg["k_conv"])
    att_left = int(cfg["att_left_ctx"])
    att_right = int(cfg["att_right_ctx"])
    att_context_size = [att_left+att_right, 0]
    conv_context_size = [k_conv-1, 0]

    nemo_enc = ConformerEncoder(
        feat_in=d_model,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ff_expansion_factor=ff_mult,
        conv_kernel_size=k_conv,
        conv_context_size=conv_context_size,
        att_context_size=att_context_size,
    )
    nemo_enc.eval()

    weights_path = os.path.join(model_dir, "model.safetensors")
    sd = load_safetensors(weights_path)
    sd = {k.strip("encoder."): v for k, v in sd.items() if "pre_encode" not in k}
    missing, unexpected = nemo_enc.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] Missing NeMo params: {len(missing)} (showing first 100): {missing[:100]}")
    if unexpected:
        print(f"[warn] Unexpected NeMo params: {len(unexpected)} (showing first 100): {unexpected[:100]}")

    return nemo_enc, {"d_model": d_model, "att_left": att_left, "att_right": att_right}


@torch.no_grad()
def _nemo_full_prefill_outputs_batched(nemo_enc: torch.nn.Module, seq_inputs: torch.Tensor) -> torch.Tensor:
    """
    seq_inputs: [B, T, D]
    returns: [B, T, D]
    """
    assert seq_inputs.dim() == 3, f"Expected [B, T, D], got {seq_inputs.shape}"
    B, T, D = seq_inputs.shape
    audio_signal = seq_inputs.contiguous()  # [B, T, D]
    length = torch.full((B,), T, dtype=torch.long, device=audio_signal.device)
    encoded, _ = nemo_enc(
        audio_signal=audio_signal,
        length=length,
        bypass_pre_encode=True,
    )
    # convert [B, D, T] to [B, T, D]
    return encoded.transpose(1, 2).contiguous()

async def _run_stream(
    worker_id: int,
    engine,
    sampling_params,
    seq_inputs_1t: torch.Tensor,   # [T, D]
    nemo_ref_1t: torch.Tensor,     # [T, D]
    first_packet_len: int = 1,
):
    from vllm.inputs.data import EmbedsPrompt

    assert seq_inputs_1t.dim() == 2 and nemo_ref_1t.dim() == 2
    STEPS, D_IN = seq_inputs_1t.shape
    req_id = f"fastconformer-correctness-w{worker_id}"

    diffs_l2: list[float] = []
    diffs_max_abs: list[float] = []

    gen_iter = engine.generate(
        request_id=req_id,
        prompt=EmbedsPrompt(prompt_embeds=seq_inputs_1t[:first_packet_len, :].contiguous()),
        sampling_params=sampling_params,
        is_streaming=True,
    )

    try:
        first_out = await gen_iter.__anext__()
        hs = first_out.outputs[0].hidden_states[-1]
        if hs.dim() == 2:
            hs = hs[-1:, :]
        ref = nemo_ref_1t[first_packet_len - 1:first_packet_len, :]
        l2 = torch.norm(hs - ref).item()
        max_abs = torch.max(torch.abs(hs - ref)).item()
        diffs_l2.append(l2)
        diffs_max_abs.append(max_abs)
        print(f"[worker {worker_id} | step {first_packet_len-1}] l2={l2:.6e} | max_abs={max_abs:.6e}")
    except StopAsyncIteration:
        raise RuntimeError(f"Worker {worker_id}: stream ended before first output")

    for i in range(first_packet_len, STEPS):
        pkt = seq_inputs_1t[i:i+1, :].contiguous()
        await engine.append_request(request_id=req_id, input_embeds=pkt)
        try:
            out = await gen_iter.__anext__()
        except StopAsyncIteration:
            break
        hs = out.outputs[0].hidden_states[-1]
        if hs.dim() == 2:
            hs = hs[-1:, :]
        ref = nemo_ref_1t[i:i+1, :]
        l2 = torch.norm(hs - ref).item()
        max_abs = torch.max(torch.abs(hs - ref)).item()
        diffs_l2.append(l2)
        diffs_max_abs.append(max_abs)
        print(f"[worker {worker_id} | step {i}] l2={l2:.6e} | max_abs={max_abs:.6e}")

    return diffs_l2, diffs_max_abs

async def main():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import SamplingParams

    parser = argparse.ArgumentParser(description="vLLM FastConformer correctness (single run)")
    parser.add_argument("--model", type=str, default="/home/scratch.jdaw_coreai/landrew/fastconformer_hf/")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--num_streams", type=int, default=2, help="Number of concurrent streams to run")
    args = parser.parse_args()

    DTYPE = _get_dtype(args.dtype)

    print(f"eager: {args.eager}")
    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

    nemo_enc, meta = _build_nemo_from_config(args.model, dtype=DTYPE)
    d_model = meta["d_model"]

    torch.manual_seed(0)
    STEPS = int(args.steps)
    D_IN = d_model
    NUM_STREAMS = args.num_streams
    seq_inputs = torch.randn(NUM_STREAMS, STEPS, D_IN, dtype=DTYPE)  # [B, T, D]

    nemo_enc = nemo_enc.to(device="cuda", dtype=DTYPE)
    with torch.no_grad():
        nemo_prefill_out = _nemo_full_prefill_outputs_batched(nemo_enc, seq_inputs.cuda()).cpu()  # [B, T, D]

    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enable_prompt_embeds=True,
        enforce_eager=args.eager,
        return_hidden_states=True,
        skip_tokenizer_init=True,
        dtype=args.dtype,
        block_size=128,
        disable_log_stats=True,
        compilation_config={"cudagraph_mode": "FULL"},
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(max_tokens=STEPS)
    first_packet_len = 1

    tasks = []
    for b in range(NUM_STREAMS):
        tasks.append(
            _run_stream(
                worker_id=b,
                engine=engine,
                sampling_params=sampling_params,
                seq_inputs_1t=seq_inputs[b],
                nemo_ref_1t=nemo_prefill_out[b],
                first_packet_len=first_packet_len,
            )
        )

    streams_results = await torch.asyncio.gather(*tasks) if hasattr(torch, "asyncio") else await __import__("asyncio").gather(*tasks)

    diffs_l2_all: list[float] = []
    diffs_max_abs_all: list[float] = []
    for (dl2, dmax) in streams_results:
        diffs_l2_all.extend(dl2)
        diffs_max_abs_all.extend(dmax)

    d = np.array(diffs_l2_all)
    m = np.array(diffs_max_abs_all)
    print("\ndiffs vs NeMo prefill (vLLM last-token hidden state vs NeMo prefill output at t) across all streams:")
    print(f"streams compared: {NUM_STREAMS}")
    print(f"total steps compared:   {len(diffs_l2_all)}")
    print(f"L2 mean:          {d.mean():.6e}")
    print(f"L2 p50/p90:       {np.percentile(d,50):.6e} / {np.percentile(d,90):.6e}")
    print(f"L2 min/max:       {d.min():.6e} / {d.max():.6e}")
    print(f"max_abs mean:     {m.mean():.6e}")
    print(f"max_abs p50/p90:  {np.percentile(m,50):.6e} / {np.percentile(m,90):.6e}")
    print(f"max_abs min/max:  {m.min():.6e} / {m.max():.6e}")

    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
