import argparse
import asyncio
import time
import uuid
import logging
from typing import Dict, Any
import numpy as np
import torch

# Suppress verbose vLLM logging
logging.getLogger("vllm").setLevel(logging.WARNING)

try:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("Error: Failed to import vllm.")
    print("Please install vllm: pip install vllm")
    exit(1)


# FastConformer constants
SAMPLES_PER_FRAME = 1280  # 80ms at 16kHz
PREEMPHASIS_COEF = 0.97


def generate_random_audio(num_frames: int) -> torch.Tensor:
    """
    Generate random audio samples for benchmarking.
    Returns audio tensor with preemphasis applied.
    """
    num_samples = num_frames * SAMPLES_PER_FRAME
    audio = torch.randn(num_samples, dtype=torch.float32)
    # Apply preemphasis
    audio = torch.cat((audio[0:1], audio[1:] - PREEMPHASIS_COEF * audio[:-1]), dim=0)
    return audio


async def run_request(
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    num_frames: int,
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    Sends a single request to the vLLM engine and records metrics.
    Processes num_frames of audio through FastConformer.
    """
    # Generate random audio for this request
    audio = generate_random_audio(num_frames)

    request_start_time = time.perf_counter()
    frame_latencies = []
    frame_count = 0

    try:
        # First frame (prefill)
        prompt_len = 1
        i = 0
        inputs = {
            "prompt_token_ids": [0] * prompt_len,
            "custom_inputs": {
                "audio": audio[
                    i * SAMPLES_PER_FRAME : (i + prompt_len) * SAMPLES_PER_FRAME
                ].view(prompt_len, SAMPLES_PER_FRAME)
            },
        }
        i += prompt_len

        frame_start_time = time.perf_counter()
        last_frame_time = None

        async for output in engine.generate(
            inputs, sampling_params=sampling_params, request_id=request_id
        ):
            now = time.perf_counter()

            # Get the acoustic embedding from custom_outputs
            acoustic_emb = output.outputs[0].custom_outputs.get("acoustic_emb")
            frame_count += 1

            if last_frame_time is not None:
                # Record inter-frame latency
                ifl = now - last_frame_time
                frame_latencies.append(ifl)
                metrics["inter_frame_latencies"].append(ifl)
            else:
                # First frame - record time-to-first-frame
                ttff = now - request_start_time
                metrics["ttff_latencies"].append(ttff)

            last_frame_time = now

            # Check if we've processed all frames
            if i >= num_frames:
                await engine.abort(request_id)
                break

            # Prepare next decode step inputs
            next_inputs = {
                "audio": audio[
                    i * SAMPLES_PER_FRAME : (i + 1) * SAMPLES_PER_FRAME
                ].unsqueeze(0)
            }
            i += 1
            await engine.append_request(
                request_id=request_id, custom_inputs=next_inputs
            )

        # After the loop finishes (sequence is done)
        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time

        # Record final metrics for this request
        metrics["request_latencies"].append(request_latency)
        metrics["completed_sequences"] += 1
        metrics["total_frames"] += frame_count

        # Calculate average frame latency for this request
        if frame_latencies:
            avg_frame_latency = np.mean(frame_latencies)
            metrics["avg_frame_latencies_per_request"].append(avg_frame_latency)

    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        import traceback

        traceback.print_exc()
        metrics["failed_sequences"] += 1


def calculate_and_print_metrics(
    metrics: Dict[str, Any], total_time: float, args: argparse.Namespace
):
    """
    Calculates and prints the final benchmark statistics.
    """
    total_sequences = metrics["completed_sequences"]
    if total_sequences == 0:
        print("Error: No sequences completed.")
        return

    total_frames = metrics["total_frames"]

    avg_seq_per_sec = total_sequences / total_time
    avg_frames_per_sec = total_frames / total_time
    avg_frames_per_seq = total_frames / total_sequences

    avg_seq_latency_s = np.mean(metrics["request_latencies"])
    p95_seq_latency_s = np.percentile(metrics["request_latencies"], 95)

    avg_ttff_ms = np.mean(metrics["ttff_latencies"]) * 1000
    p95_ttff_ms = np.percentile(metrics["ttff_latencies"], 95) * 1000

    avg_ifl_ms = np.mean(metrics["inter_frame_latencies"]) * 1000
    p95_ifl_ms = np.percentile(metrics["inter_frame_latencies"], 95) * 1000

    avg_frame_latency_ms = np.mean(metrics["avg_frame_latencies_per_request"]) * 1000

    # Real-time factor: processing time vs audio duration
    # Each frame is 80ms of audio
    audio_duration_per_seq = args.num_frames * 0.08  # seconds
    rtf = avg_seq_latency_s / audio_duration_per_seq

    print("\n--- FastConformer Benchmark Results ---")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Frames per request: {args.num_frames} ({args.num_frames * 80}ms audio)")
    print("---")
    print(f"Total duration: {total_time:.2f} s")
    print(f"Total completed sequences: {total_sequences}")
    print(f"Total failed sequences: {metrics['failed_sequences']}")
    print(f"Total frames processed: {total_frames}")
    print(f"Average frames per sequence: {avg_frames_per_seq:.2f}")
    print("--- Throughput ---")
    print(f"Average sequences/sec: {avg_seq_per_sec:.2f}")
    print(f"Average frames/sec: {avg_frames_per_sec:.2f}")
    print(f"Real-time factor (RTF): {rtf:.4f}x (< 1.0 means faster than real-time)")
    print("--- Latency ---")
    print(f"Average sequence latency: {avg_seq_latency_s:.4f} s")
    print(f"P95 sequence latency: {p95_seq_latency_s:.4f} s")
    print(f"Average TTFF (Time-To-First-Frame): {avg_ttff_ms:.2f} ms")
    print(f"P95 TTFF: {p95_ttff_ms:.2f} ms")
    print(f"Average IFL (Inter-Frame Latency): {avg_ifl_ms:.2f} ms")
    print(f"P95 IFL: {p95_ifl_ms:.2f} ms")
    print(f"Average frame latency per worker: {avg_frame_latency_ms:.2f} ms")
    print("---------------------------------------")


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    num_frames: int,
    metrics: Dict[str, Any],
):
    """
    A persistent worker that continuously sends requests until
    the global request counter reaches zero.
    """
    while True:
        # Atomically check and decrement the request counter
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                # All requests have been assigned, worker can exit
                break
            metrics["requests_to_run"] -= 1

        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        # Run the request *outside* the lock
        await run_request(engine, sampling_params, num_frames, metrics, request_id)


def init_metrics(num_requests: int):
    return {
        "request_latencies": [],
        "inter_frame_latencies": [],
        "ttff_latencies": [],
        "avg_frame_latencies_per_request": [],
        "completed_sequences": 0,
        "failed_sequences": 0,
        "total_frames": 0,
        "requests_to_run": num_requests,
        "lock": asyncio.Lock(),
    }


async def main():
    parser = argparse.ArgumentParser(
        description="vLLM FastConformer Benchmarking Script"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to run with torch profiler",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Whether to skip warmup",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent workers (N)",
    )
    parser.add_argument(
        "-m",
        "--num-requests",
        type=int,
        default=512,
        help="Total number of requests to send (M)",
    )
    parser.add_argument(
        "-f",
        "--num-frames",
        type=int,
        default=100,
        help="Number of audio frames per request (each frame is 80ms)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model length (context size)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.8,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./fastconformer_model/",
        help="Path to FastConformer model",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for KV cache",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        choices=["auto", "dummy"],
        help="Load format: 'auto' for real weights, 'dummy' for random weights",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graphs)",
    )
    args = parser.parse_args()

    print("Starting FastConformer benchmark...")
    print(f"Concurrency: {args.concurrency}, Num Requests: {args.num_requests}")
    print(f"Frames per request: {args.num_frames} ({args.num_frames * 80}ms audio)")
    print(f"Max Model Len: {args.max_model_len}")

    # 1. Create Engine Args
    engine_args_kwargs = {
        "model": args.model_path,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "block_size": args.block_size,
        "skip_tokenizer_init": True,
        "enable_prefix_caching": False,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.load_format == "dummy":
        engine_args_kwargs["load_format"] = "dummy"

    engine_args = AsyncEngineArgs(**engine_args_kwargs)

    # 2. Create Engine
    engine = AsyncLLM.from_engine_args(engine_args)
    if args.profile:
        await engine.start_profile()

    # 3. Create Sampling Params
    # max_tokens should be at least num_frames to allow all outputs
    sampling_params = SamplingParams(
        max_tokens=args.num_frames + 10,
        skip_sampling=True,
    )

    # Run warmup and then actual benchmark
    warmup_num = 0 if args.no_warmup else 3 * args.concurrency 
    for run, num_requests in enumerate([warmup_num, args.num_requests]):
        if num_requests == 0:
            continue
        metrics = init_metrics(num_requests)

        run_name = "Warmup" if run == 0 else "Benchmark"
        print(f"\n--- Starting {run_name} run for {num_requests} requests ---")
        start_time = time.perf_counter()

        # Create and start C worker tasks
        tasks = []
        for i in range(args.concurrency):
            tasks.append(
                asyncio.create_task(
                    worker(
                        worker_id=i,
                        engine=engine,
                        sampling_params=sampling_params,
                        num_frames=args.num_frames,
                        metrics=metrics,
                    )
                )
            )

        # Wait for all worker tasks to finish
        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"{run_name} finished. Total time: {total_time:.2f}s")

        if run > 0:
            if args.profile:
                await engine.stop_profile()
            # Calculate and print final metrics
            calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
