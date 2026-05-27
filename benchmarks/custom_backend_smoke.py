# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test helper for CUSTOM attention backend benchmarking.

Usage (example):
  python benchmarks/custom_backend_smoke.py --model <model_path> --prompt "Hello"

This script runs a short throughput benchmark for both CUSTOM and FLASH_ATTN
backends and prints the commands it uses so results can be compared.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


def _run(cmd: list[str]) -> int:
    print(f"[bench] {shlex.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom backend smoke bench")
    parser.add_argument("--model", required=True, help="Model path or repo id")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    base = [
        "python",
        "-m",
        "vllm.benchmarks.throughput",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--num-prompts",
        str(args.num_prompts),
        "--max-tokens",
        str(args.max_tokens),
    ]

    print("[bench] Running CUSTOM backend...")
    custom_cmd = base + ["--attention-backend", "CUSTOM"]
    rc_custom = _run(custom_cmd)

    print("[bench] Running FLASH_ATTN backend...")
    flash_cmd = base + ["--attention-backend", "FLASH_ATTN"]
    rc_flash = _run(flash_cmd)

    return rc_custom or rc_flash


if __name__ == "__main__":
    raise SystemExit(main())
