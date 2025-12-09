import argparse
import os

import numpy as np
import torch


def _get_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


async def main():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import SamplingParams

    parser = argparse.ArgumentParser(description="vLLM FastConformer comparison with NeMo outputs")
    parser.add_argument("--model", type=str, default="/home/scratch.jdaw_coreai/landrew/fastconformer_hf/")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--input-dir", type=str, default="./fastconformer_test_data")
    args = parser.parse_args()

    DTYPE = _get_dtype(args.dtype)

    inputs_path = os.path.join(args.input_dir, "inputs.pt")
    outputs_path = os.path.join(args.input_dir, "nemo_outputs.pt")

    print(f"Loading inputs from {inputs_path}...")
    seq_inputs = torch.load(inputs_path, weights_only=True)

    print(f"Loading NeMo outputs from {outputs_path}...")
    nemo_prefill_out = torch.load(outputs_path, weights_only=True)

    STEPS = seq_inputs.shape[0]
    print(f"Loaded {STEPS} steps, input shape: {seq_inputs.shape}, output shape: {nemo_prefill_out.shape}")

    print(f"eager: {args.eager}")
    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        #enable_prompt_embeds=True,
        enforce_eager=args.eager,
        #return_hidden_states=True,
        skip_tokenizer_init=True,
        dtype=args.dtype,
        block_size=128,
        disable_log_stats=True,
        compilation_config={"cudagraph_mode": "FULL"},
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    req_id = "fastconformer-correctness"
    first_packet_len = 1

    diffs_l2: list[float] = []
    diffs_max_abs: list[float] = []

    first_packet = seq_inputs[:first_packet_len, :].contiguous()
    sampling_params = SamplingParams(max_tokens=STEPS, skip_sampling=True)
    prefill_inputs = {
        "prompt_token_ids": [0] * first_packet_len,
        "custom_inputs": {"proc_melspec": first_packet},
    }
    gen_iter = engine.generate(
        prefill_inputs,
        sampling_params=sampling_params,
        request_id=req_id,
    )

    try:
        first_out = await gen_iter.__anext__()
        hs = first_out.outputs[0].custom_outputs["acoustic_emb"]
        if hs.dim() == 2:
            hs = hs[-1:, :]
        ref = nemo_prefill_out[first_packet_len - 1:first_packet_len, :]
        l2 = torch.norm(hs - ref).item()
        max_abs = torch.max(torch.abs(hs - ref)).item()
        diffs_l2.append(l2)
        diffs_max_abs.append(max_abs)
        print(f"[compare step {first_packet_len-1}] l2={l2:.6e} | max_abs={max_abs:.6e}")
    except StopAsyncIteration:
        print("Stream ended before first output")
        return

    for i in range(first_packet_len, STEPS):
        pkt = seq_inputs[i:i+1, :].contiguous()
        await engine.append_request(request_id=req_id, custom_inputs={"proc_melspec": pkt})
        try:
            out = await gen_iter.__anext__()
        except StopAsyncIteration:
            break

        hs = out.outputs[0].custom_outputs["acoustic_emb"]
        if hs.dim() == 2:
            hs = hs[-1:, :]
        ref = nemo_prefill_out[i:i+1, :]
        l2 = torch.norm(hs - ref).item()
        max_abs = torch.max(torch.abs(hs - ref)).item()
        diffs_l2.append(l2)
        diffs_max_abs.append(max_abs)
        print(f"[compare step {i}] l2={l2:.6e} | max_abs={max_abs:.6e}")

    d = np.array(diffs_l2)
    m = np.array(diffs_max_abs)
    print("diffs vs NeMo prefill (vLLM last-token hidden state vs NeMo prefill output at t):")
    print(f"steps compared:   {len(diffs_l2)}")
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

