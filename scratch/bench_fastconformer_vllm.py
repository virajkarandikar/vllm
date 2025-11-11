import asyncio
import argparse
import time

import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import EmbedsPrompt


async def main():
    parser = argparse.ArgumentParser(description="FastConformer vLLM benchmarking")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the input")
    parser.add_argument("--steps", type=int, default=100, help="Number of decode steps")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Number of warmup decode steps (not measured)")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler")
    args = parser.parse_args()

    # not used. asynchronous engine only supports batch size 1
    batch_size = args.batch_size
    STEPS = args.steps
    WARMUP_STEPS = args.warmup_steps
    TOTAL_STEPS = 100 + WARMUP_STEPS + STEPS

    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

    engine_args = AsyncEngineArgs(
        model="/home/scratch.jdaw_coreai/landrew/fastconformer_hf/",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        block_size=128,
        enable_prompt_embeds=True,
        enforce_eager=False,
        return_hidden_states=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        compilation_config={"level": 0, "cudagraph_mode": "FULL"}
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    if args.profile:
        await engine.start_profile()

    req_id = "latency-bench"
    latency_measurements: list[float] = []

    torch.manual_seed(0)
    D_IN = engine.model_config.get_hidden_size()
    print(f"D_IN: {D_IN}")
    seq_inputs = torch.randn(1, TOTAL_STEPS, D_IN)
    # shape: [1, TOTAL_STEPS, D_IN]
    first_packet = seq_inputs[:, :1, :].contiguous()

    gen_iter = engine.generate(
        request_id=req_id,
        prompt=EmbedsPrompt(prompt_embeds=first_packet.reshape(-1, D_IN)),
        sampling_params=SamplingParams(max_tokens=TOTAL_STEPS),
        is_streaming=True,
    )

    async def handle(output, step, latency=None):
        if latency is not None:
            print(f"[step {step}] -> latency: {latency*1000:.2f} ms")
        else:
            print(f"[step {step}] -> done")

    try:
        first_out = await gen_iter.__anext__()
        await handle(first_out, 0, latency=None)
    except StopAsyncIteration as e:
        print(f"StopAsyncIteration: {e}")
        pass

    for i in range(1, TOTAL_STEPS):
        pkt = seq_inputs[:, i:(i+1), :].contiguous()
        pkt = pkt.reshape(-1, D_IN).contiguous()
        await engine.append_request(request_id=req_id, input_embeds=pkt)
        try:
            if i <= WARMUP_STEPS:
                out = await gen_iter.__anext__()
                await handle(out, i, latency=None)
            else:
                t0 = time.perf_counter()
                out = await gen_iter.__anext__()
                t1 = time.perf_counter()
                latency = t1 - t0
                latency_measurements.append(latency)
                await handle(out, i, latency=latency)
        except StopAsyncIteration as e:
            print(f"StopAsyncIteration: {e}")
            break

    if args.profile:
        await engine.stop_profile()

    if latency_measurements:
        import numpy as np
        arr = np.array(latency_measurements)
        mean_ms = arr.mean() * 1000.0
        std_ms = arr.std() * 1000.0
        p50_ms = np.percentile(arr, 50) * 1000.0

        print("=== Benchmark results ===")
        print(f"   Engine: vllm")
        print(f"   Total frames: {STEPS}")
        print(f"   Feature dim: {D_IN}")
        print(f"   Batch size: {batch_size}")
        print(f"   Mode: {'eager' if engine_args.enforce_eager else 'graph'}")
        print(f"   Inter-token latency: {p50_ms:.3f} ms")
        print(f"   Mean: {mean_ms:.3f} ms, Std: {std_ms:.3f} ms")

    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

if __name__ == "__main__":
    asyncio.run(main())
