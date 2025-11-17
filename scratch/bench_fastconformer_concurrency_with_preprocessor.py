import argparse
import asyncio
import time
import uuid
from typing import Dict, Any
import numpy as np
import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.inputs.data import EmbedsPrompt
from vllm.model_executor.models.fastconformer_preprocessor import FastConformerPreprocessor

async def run_request(
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_embed_dim: int,
    steps: int,
    metrics: Dict[str, Any],
    request_id: str
):
    """
    Sends a single FastConformer-style streaming request.
    Runs the FastConformer preprocessor per step to produce embeddings, and
    records TTFT, inter-step latencies, and total request latency.
    """
    request_start_time = time.perf_counter()
    last_step_time = None
    step_count = 0

    try:
        preprocessor = FastConformerPreprocessor().cuda()
        preprocessor.capture_cuda_graph()

        SAMPLES_PER_STEP = 160 * 8
        audio_stream = torch.randn(1, steps * SAMPLES_PER_STEP, device="cuda", dtype=torch.float32)

        def build_preproc_input(step_idx: int) -> torch.Tensor:
            start = step_idx * SAMPLES_PER_STEP
            end = start + SAMPLES_PER_STEP
            return audio_stream[:, start:end].contiguous()

        @torch.no_grad()
        def run_preproc_forward(chunk_b_t: torch.Tensor) -> torch.Tensor:
            feat_seq = preprocessor.forward_cuda_graph(chunk_b_t)
            frame = feat_seq[:, -1, :].contiguous()
            # NOTE: vLLM requires CPU embeddings; potential performance hit here
            return frame.cpu().to(torch.float32)

        first_chunk = build_preproc_input(0)
        first_packet = run_preproc_forward(first_chunk)
        gen_iter = engine.generate(
            request_id=request_id,
            prompt=EmbedsPrompt(prompt_embeds=first_packet),
            sampling_params=sampling_params,
            is_streaming=True,
        )

        try:
            first_out = await gen_iter.__anext__()
            now = time.perf_counter()
            ttft = now - request_start_time
            metrics["ttft_latencies"].append(ttft)
            last_step_time = now
            step_count += 1
        except StopAsyncIteration:
            raise RuntimeError("Stream ended before first output")

        for i in range(1, steps):
            t0 = time.perf_counter()
            chunk = build_preproc_input(i)
            pkt = run_preproc_forward(chunk)
            await engine.append_request(request_id=request_id, input_embeds=pkt)
            try:
                _ = await gen_iter.__anext__()
            except StopAsyncIteration:
                break
            t1 = time.perf_counter()
            latency = t1 - t0
            metrics["inter_step_latencies"].append(latency)
            last_step_time = t1
            step_count += 1

        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time
        metrics["request_latencies"].append(request_latency)
        metrics["completed_sequences"] += 1
        metrics["total_steps"] += step_count

    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        metrics["failed_sequences"] += 1


def calculate_and_print_metrics(
    metrics: Dict[str, Any],
    total_time: float,
    args: argparse.Namespace
):
    """
    Calculates and prints the final benchmark statistics for FastConformer embeddings.
    """
    total_sequences = metrics["completed_sequences"]
    if total_sequences == 0:
        print("Error: No sequences completed.")
        return
    
    total_steps = metrics["total_steps"]
    
    avg_seq_per_sec = total_sequences / total_time
    avg_step_per_sec = total_steps / total_time
    avg_seq_len = total_steps / total_sequences if total_sequences > 0 else 0.0
    
    avg_seq_latency_s = float(np.mean(metrics["request_latencies"])) if metrics["request_latencies"] else 0.0
    p95_seq_latency_s = float(np.percentile(metrics["request_latencies"], 95)) if metrics["request_latencies"] else 0.0
    
    avg_ttft_ms = float(np.mean(metrics["ttft_latencies"]) * 1000) if metrics["ttft_latencies"] else 0.0
    p95_ttft_ms = float(np.percentile(metrics["ttft_latencies"], 95) * 1000) if metrics["ttft_latencies"] else 0.0

    avg_isl_ms = float(np.mean(metrics["inter_step_latencies"]) * 1000) if metrics["inter_step_latencies"] else 0.0
    p95_isl_ms = float(np.percentile(metrics["inter_step_latencies"], 95) * 1000) if metrics["inter_step_latencies"] else 0.0

    print("\n--- vLLM FastConformer Benchmark Results ---")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Steps per request: {args.steps}")
    print("---")
    print(f"Total duration: {total_time:.2f} s")
    print(f"Total completed sequences: {total_sequences}")
    print(f"Total failed sequences: {metrics['failed_sequences']}")
    print(f"Total steps processed: {total_steps}")
    print(f"Average sequence length: {avg_seq_len:.2f} steps")
    print("--- Throughput ---")
    print(f"Average sequences/sec: {avg_seq_per_sec:.2f}")
    print(f"Average steps/sec: {avg_step_per_sec:.2f}")
    print("--- Latency ---")
    print(f"Average sequence latency: {avg_seq_latency_s:.2f} s")
    print(f"P95 sequence latency: {p95_seq_latency_s:.2f} s")
    print(f"Average TTFT: {avg_ttft_ms:.2f} ms")
    print(f"P95 TTFT: {p95_ttft_ms:.2f} ms")
    print(f"Average inter-step latency: {avg_isl_ms:.2f} ms")
    print(f"P95 inter-step latency: {p95_isl_ms:.2f} ms")
    print("------------------------------")


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_embed_dim: int,
    steps: int,
    metrics: Dict[str, Any],
):
    """
    A persistent worker that continuously sends requests until
    the global request counter reaches zero.
    """
    while True:
        
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                break
            metrics["requests_to_run"] -= 1
        
        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        await run_request(
            engine,
            sampling_params,
            input_embed_dim,
            steps,
            metrics,
            request_id,
        )

async def main():
    parser = argparse.ArgumentParser(description="vLLM FastConformer Benchmarking Script (embeddings)")
    parser.add_argument(
        "-c", "--concurrency", type=int, default=16,
        help="Number of concurrent workers (N)"
    )
    parser.add_argument(
        "-m", "--num-requests", type=int, default=512,
        help="Total number of requests to send (M)"
    )
    parser.add_argument(
        "--steps", type=int, default=80,
        help="Number of streaming steps per request"
    )
    parser.add_argument(
        "--model", type=str, default="/home/scratch.jdaw_coreai/landrew/fastconformer_hf/",
        help="Path/name of the FastConformer model bundle"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum model length (context size)"
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.8,
        help="GPU memory utilization (0.0 to 1.0)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        help="Model dtype (e.g., bfloat16, float16)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable PyTorch Profiler"
    )
    args = parser.parse_args()

    print("syncing cuda...")
    torch.cuda.synchronize()
    print("cuda synced")

    print("Starting vLLM FastConformer embeddings benchmark...")
    print(f"Model: {args.model}, Concurrency: {args.concurrency}, Num Requests: {args.num_requests}")
    print(f"Steps: {args.steps}, Max Model Len: {args.max_model_len}")
    
    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prompt_embeds=True,
        enforce_eager=False,
        return_hidden_states=True,
        skip_tokenizer_init=True,
        dtype=args.dtype,
        block_size=128,
        disable_log_stats=True,
        compilation_config={"cudagraph_mode": "FULL"}
    )

    engine = AsyncLLM.from_engine_args(engine_args)

    if args.profile:
        await engine.start_profile()

    sampling_params = SamplingParams(
        max_tokens=args.steps,
    )
    
    metrics = {
        "request_latencies": [],
        "inter_step_latencies": [],
        "ttft_latencies": [],
        "completed_sequences": 0,
        "failed_sequences": 0,
        "total_steps": 0,
        "requests_to_run": args.num_requests,
        "lock": asyncio.Lock(),
    }

    print(f"\nStarting benchmark for {args.num_requests} requests... This may take a moment.")
    start_time = time.perf_counter()

    tasks = []
    for i in range(args.concurrency):
        tasks.append(
            asyncio.create_task(
                worker(
                    worker_id=i,
                    engine=engine,
                    sampling_params=sampling_params,
                    input_embed_dim=512,
                    steps=args.steps,
                    metrics=metrics,
                )
            )
        )
    
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()

    total_time = end_time - start_time

    if args.profile:
        await engine.stop_profile()
    
    print(f"\nBenchmark finished. Total time: {total_time:.2f}s")

    calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
