import asyncio
import json
import os
import shutil
import time
from contextlib import contextmanager

import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import EmbedsPrompt


async def main():
    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

    STEPS = 1000

    engine_args = AsyncEngineArgs(
        model="gpt2",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        block_size=128,
        enable_prompt_embeds=True,
        enforce_eager=True,
        return_hidden_states=True,
        skip_tokenizer_init=True,
        dtype="bfloat16"
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    req_id = "latency-bench"
    latency_measurements: list[float] = []

    torch.manual_seed(0)
    D_IN = engine.model_config.get_hidden_size()
    seq_inputs = torch.randn(1, STEPS, D_IN)
    first_packet_len = 1
    first_packet = seq_inputs[0, :first_packet_len, :].contiguous()

    gen_iter = engine.generate(
        request_id=req_id,
        prompt=EmbedsPrompt(prompt_embeds=first_packet),
        sampling_params=SamplingParams(max_tokens=STEPS),
        is_streaming=True,
    )

    async def handle(output, step, latency=None):
        # hs = output.outputs[0].hidden_states[-1]
        # if hs.dim() == 2:
        #     hs = hs[-1:, :]
        # if hs is None or hs.numel() == 0:
        #     raise RuntimeError("Hidden states are None or empty")
        # if latency is not None:
        #     print(f"[step {step}] -> hs shape: {hs.shape} | latency: {latency*1000:.2f} ms")
        # else:
        #     print(f"[step {step}] -> hs shape: {hs.shape}")
        if latency is not None:
            print(f"[step {step}] -> latency: {latency*1000:.2f} ms")
        else:
            print(f"[step {step}] -> done")

    # First token
    try:
        first_out = await gen_iter.__anext__()
        await handle(first_out, first_packet_len - 1, latency=None)
    except StopAsyncIteration as e:
        print(f"StopAsyncIteration: {e}")
        pass

    for i in range(first_packet_len, STEPS):
        pkt = seq_inputs[:, i, :].contiguous()
        await engine.append_request(request_id=req_id, input_embeds=pkt)
        try:
            t0 = time.perf_counter()
            out = await gen_iter.__anext__()
            t1 = time.perf_counter()
            latency = t1 - t0
            latency_measurements.append(latency)
            await handle(out, i, latency=latency)
        except StopAsyncIteration as e:
            print(f"StopAsyncIteration: {e}")
            break

    if latency_measurements:
        import numpy as np
        arr = np.array(latency_measurements)
        print("latency...:")
        print(f"steps measured: {len(latency_measurements)}")
        print(f"mean latency:    {arr.mean()*1000:.2f} ms")
        print(f"p50 latency:     {np.percentile(arr,50)*1000:.2f} ms")
        print(f"p90 latency:     {np.percentile(arr,90)*1000:.2f} ms")
        print(f"min/max latency: {arr.min()*1000:.2f} / {arr.max()*1000:.2f} ms")

    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

if __name__ == "__main__":
    asyncio.run(main())
