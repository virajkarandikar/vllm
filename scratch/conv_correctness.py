import asyncio
import json
import os
import shutil
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import EmbedsPrompt

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution
from safetensors.torch import load_file as load_safetensors

BUNDLE_DIR = "/home/scratch.jdaw_coreai/landrew/fastconformer_hf/"
TMP_DIR = "/home/scratch.jdaw_coreai/landrew/conv_correctness_tmp/"
D_IN = 512
DTYPE = torch.float32

def _get_dtype_str(dtype: torch.dtype) -> str:
    match dtype:
        case torch.float32:
            return "float32"
        case torch.bfloat16:
            return "bfloat16"
        case _:
            raise ValueError(f"unsupported dtype: {dtype}")

@contextmanager
def use_tmp_bundle_dir(bundle_dir, tmp_dir):
    def _force_rmtree(path, max_tries=5, delay=0.5):
        last_ex = None
        for _ in range(max_tries):
            try:
                shutil.rmtree(path)
                return
            except OSError as ex:
                last_ex = ex
                time.sleep(delay)
        raise RuntimeError(f"Failed to remove tmp_dir '{path}': {last_ex}") from last_ex

    if os.path.exists(tmp_dir):
        _force_rmtree(tmp_dir)
    shutil.copytree(bundle_dir, tmp_dir)
    cfg_path = os.path.join(bundle_dir, "config.json")
    out_cfg_path = os.path.join(tmp_dir, "config.json")
    cfg = json.load(open(cfg_path, "r"))
    cfg["conv_only"] = True
    cfg["hidden_size"] = D_IN
    cfg["d_model"] = D_IN
    with open(out_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    try:
        print(f"using tmp bundle dir: {tmp_dir}")
        yield tmp_dir
    finally:
        print(f"removing tmp bundle dir: {tmp_dir}")
        if os.path.exists(tmp_dir):
            try:
                _force_rmtree(tmp_dir)
            except Exception as ex:
                print(f"[warn] Could not remove tmp bundle dir {tmp_dir}: {ex}")
        print(f"removed tmp bundle dir: {tmp_dir}")


async def main():
    with use_tmp_bundle_dir(BUNDLE_DIR, TMP_DIR) as model_dir:
        STEPS = 100

        engine_args = AsyncEngineArgs(
            model=model_dir,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            block_size=128,
            enable_prompt_embeds=True,
            enforce_eager=True,
            return_hidden_states=True,
            skip_tokenizer_init=True,
            dtype=_get_dtype_str(DTYPE)
        )
        engine = AsyncLLM.from_engine_args(engine_args)

        req_id = "conv-correctness"
        emitted_ids: list[int] = []

        latency_measurements: list[float] = []

        torch.manual_seed(0)
        seq_inputs = torch.randn(1, STEPS, D_IN, dtype=DTYPE)
        first_packet_len = 1
        first_packet = seq_inputs[0, :first_packet_len, :].contiguous()

        cfg = json.load(open(os.path.join(model_dir, "config.json"), "r"))
        d_model = int(cfg.get("d_model", D_IN))

        if d_model != D_IN:
            print(f"[warn] cfg d_model ({d_model}) != D_IN ({D_IN}); using d_model={d_model}")

        weights_path = os.path.join(model_dir, "model.safetensors")
        sd = load_safetensors(weights_path)

        k_conv = int(cfg.get("k_conv", 9))
        nemo_conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=k_conv,
            norm_type='batch_norm',
            conv_context_size=[k_conv - 1, 0],
            use_bias=True,
        ).to(seq_inputs.dtype)
        nemo_conv.to(DTYPE).eval()

        pw1_w_key = "encoder.layers.0.conv.pointwise_conv1.weight"
        pw1_b_key = "encoder.layers.0.conv.pointwise_conv1.bias"
        dw_w_key  = "encoder.layers.0.conv.depthwise_conv.weight"
        dw_b_key  = "encoder.layers.0.conv.depthwise_conv.bias"
        bn_w_key  = "encoder.layers.0.conv.batch_norm.weight"
        bn_b_key  = "encoder.layers.0.conv.batch_norm.bias"
        bn_bt_key = "encoder.layers.0.conv.batch_norm.num_batches_tracked"
        pw2_w_key = "encoder.layers.0.conv.pointwise_conv2.weight"
        pw2_b_key = "encoder.layers.0.conv.pointwise_conv2.bias"

        with torch.no_grad():
            nemo_conv.pointwise_conv1.weight.copy_(sd[pw1_w_key])
            nemo_conv.pointwise_conv1.bias.copy_(sd[pw1_b_key])

            nemo_conv.depthwise_conv.weight.copy_(sd[dw_w_key])
            nemo_conv.depthwise_conv.bias.copy_(sd[dw_b_key])

            nemo_conv.batch_norm.weight.copy_(sd[bn_w_key])
            nemo_conv.batch_norm.bias.copy_(sd[bn_b_key])
            if hasattr(nemo_conv.batch_norm, 'num_batches_tracked') and bn_bt_key in sd:
                nemo_conv.batch_norm.num_batches_tracked.copy_(sd[bn_bt_key])

            nemo_conv.pointwise_conv2.weight.copy_(sd[pw2_w_key])
            nemo_conv.pointwise_conv2.bias.copy_(sd[pw2_b_key])

        with torch.no_grad():
            nemo_prefill_out = nemo_conv(seq_inputs)  # [1, STEPS, d_model]

        gen_iter = engine.generate(
            request_id=req_id,
            prompt=EmbedsPrompt(prompt_embeds=first_packet),
            sampling_params=SamplingParams(max_tokens=STEPS),
            is_streaming=True,
        )

        async def handle(output, step, latency=None):
            nonlocal emitted_ids
            hs = output.outputs[0].hidden_states[-1]
            if hs.dim() == 2:
                hs = hs[-1:, :]
            if hs is None or hs.numel() == 0:
                raise RuntimeError("Hidden states are None or empty")
            if latency is not None:
                print(f"[step {step}] -> hs shape: {hs.shape} | latency: {latency*1000:.2f} ms")
            else:
                print(f"[step {step}] -> hs shape: {hs.shape}")

        diffs: list[float] = []
        max_abs_diffs: list[float] = []

        try:
            first_out = await gen_iter.__anext__()
            await handle(first_out, first_packet_len - 1, latency=None)
            with torch.no_grad():
                hs = first_out.outputs[0].hidden_states[-1]
                if hs.dim() == 2:
                    hs = hs[-1:, :]
                ref = nemo_prefill_out[:, first_packet_len - 1, :]
                l2 = torch.norm(hs - ref).item()
                max_abs = torch.max(torch.abs(hs - ref)).item()
                diffs.append(l2)
                max_abs_diffs.append(max_abs)
                print(f"[compare step {first_packet_len-1}] l2={l2:.6e} | max_abs={max_abs:.6e}")
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
                # compare
                with torch.no_grad():
                    hs = out.outputs[0].hidden_states[-1]
                    ref = nemo_prefill_out[:, i, :]
                    l2 = torch.norm(hs - ref).item()
                    max_abs = torch.max(torch.abs(hs - ref)).item()
                    diffs.append(l2)
                    max_abs_diffs.append(max_abs)
                    print(f"[compare step {i}] l2={l2:.6e} | max_abs={max_abs:.6e}")
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

        if 'diffs' in locals() and diffs:
            import numpy as np
            d = np.array(diffs)
            m = np.array(max_abs_diffs)
            print("diffs vs NeMo prefill (vLLM last-token hidden state vs NeMo prefill output at t):")
            print(f"steps compared:   {len(diffs)}")
            print(f"L2 mean:          {d.mean():.6e}")
            print(f"L2 p50/p90:       {np.percentile(d,50):.6e} / {np.percentile(d,90):.6e}")
            print(f"L2 min/max:       {d.min():.6e} / {d.max():.6e}")
            print(f"max_abs mean:     {m.mean():.6e}")
            print(f"max_abs p50/p90:  {np.percentile(m,50):.6e} / {np.percentile(m,90):.6e}")
            print(f"max_abs min/max:  {m.min():.6e} / {m.max():.6e}")

if __name__ == "__main__":
    asyncio.run(main())
