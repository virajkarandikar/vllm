# fastconformer demo in vLLM

import asyncio
import json
import os
from typing import List

import torch
import sentencepiece as spm
from safetensors.torch import load_file

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import EmbedsPrompt

BUNDLE_DIR = "/home/scratch.jdaw_coreai/landrew/fastconformer_hf/"
# TODO: seems like config has right_context=1
# assume we can just do total_context=left_context+right_context=71 frames
RIGHT_CONTEXT = 0
STEPS = 80
SEED = 0

# this is hardcoded by inspecting the config
CTC_W_KEY = "ctc_decoder.decoder_layers.0.weight"
CTC_B_KEY = "ctc_decoder.decoder_layers.0.bias"

def ctc_collapse(ids: List[int], blank_id: int) -> List[int]:
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank_id:
            out.append(i)
        prev = i
    return out

async def main():
    torch.manual_seed(SEED)

    cfg = json.load(open(os.path.join(BUNDLE_DIR, "config.json"), "r"))
    blank_id  = int(cfg["blank_id"])
    mels = 80

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(BUNDLE_DIR, cfg["tokenizer"]["path"]))

    sd = load_file(os.path.join(BUNDLE_DIR, "model.safetensors"))
    if CTC_W_KEY not in sd or CTC_B_KEY not in sd:
        raise RuntimeError(f"CTC keys not found: {CTC_W_KEY}, {CTC_B_KEY}")
    W = sd[CTC_W_KEY]            # [V, D, 1]
    b = sd[CTC_B_KEY].float()    # [V]
    if W.ndim != 3 or W.shape[2] != 1:
        raise RuntimeError(f"Unexpected CTC shapes: W={tuple(W.shape)}")
    W = W.squeeze(-1).float()    # [V, D]

    engine_args = AsyncEngineArgs(
        model=BUNDLE_DIR,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prompt_embeds=True,
        enforce_eager=True,
        return_hidden_states=True,
        skip_tokenizer_init=True
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    D_IN = mels * 8

    req_id = "asr-ctc-demo"
    emitted_ids: List[int] = []

    first_packet = torch.randn(1, D_IN)
    gen_iter = engine.generate(
        request_id=req_id,
        prompt=EmbedsPrompt(prompt_embeds=first_packet),
        sampling_params=SamplingParams(max_tokens=STEPS),
        is_streaming=True,
    )

    async def handle(output,step,compute_next=False):
        nonlocal emitted_ids
        hs = output.outputs[0].hidden_states[-1]  # [1,D]
        if hs is None or hs.numel() == 0:
            raise RuntimeError("Hidden states are None or empty")
        if compute_next:
            # TODO: doesn't work with random inputs or bugged?
            # CTC projection: logits = hs @ W^T + b
            logits = (hs.float() @ W.t()) + b    # [T_new, V]
            ids = logits.argmax(dim=-1).tolist() # use greedy here
            emitted_ids.extend(ids)
            partial = sp.decode_ids(ctc_collapse(emitted_ids, blank_id)) if emitted_ids else ""
            print(f"[step {step}] +{len(ids)} frames -> partial: {partial!r}")
        else:
            print(f"[step {step}] -> hs shape: {hs.shape}")

    try:
        first_out = await gen_iter.__anext__()
        await handle(first_out,0)
    except StopAsyncIteration as e:
        print(f"StopAsyncIteration: {e}")
        pass

    for i in range(1, STEPS):
        pkt = torch.randn(1, D_IN)
        await engine.append_request(request_id=req_id, input_embeds=pkt)
        try:
            out = await gen_iter.__anext__()
            await handle(out,i)
        except StopAsyncIteration as e:
            print(f"StopAsyncIteration: {e}")
            break

    final_text = sp.decode_ids(ctc_collapse(emitted_ids, blank_id)) if emitted_ids else ""
    print("\n=== FINAL ===")
    print("emitted_frames:", len(emitted_ids))
    print("decoded:", repr(final_text))

if __name__ == "__main__":
    asyncio.run(main())
