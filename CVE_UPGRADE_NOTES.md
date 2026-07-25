# vLLM Fork CVE Upgrade Notes

---
## 0.24.0 upgrade (S2S scan, 2026-07) — 3 HIGH DoS CVEs

The `voicechat_v0.22.0` branch (fork commit `2c089ab05`, currently used by
riva-speech `docker/Dockerfile.s2s`) is flagged for three HIGH DoS CVEs, all fixed in
upstream **vLLM 0.24.0**. These are metadata-version findings — patching source does not
clear them, only a real bump to 0.24.0 does.

| CVE | GHSA | Vector (reachable only if feature enabled) |
|-----|------|--------------------------------------------|
| CVE-2026-54234 | GHSA-8wr5-jm2h-8r4f | spec-decode overlapping Generate/Abort → out-of-vocab token → GPU assert crash |
| CVE-2026-55514 | GHSA-33cg-gxv8-3p8g | M-RoPE model + `prompt_embeds` without token ids → assertion crash |
| CVE-2026-55574 | GHSA-rwxx-mrjm-wc2m | ReDoS via `structured_outputs.regex` (xgrammar/outlines, no compile timeout) |

### DONE: branch `voicechat_v0.24.0` = 146 original commits rebased onto v0.24.0

The current `voicechat_v0.24.0` branch replays all **146 original fork commits** (verbatim, authors
and messages preserved) onto upstream `v0.24.0`, followed by ONE `reconcile ...` commit that sets the
final tree exactly equal to the validated result. History is linear (0 merges). Produced via:
`git rebase --onto v0.24.0 v0.22.0 -X theirs` (auto-resolve toward the incoming fork commit), then
`git checkout netdiff-validated -- .` + commit to reconcile. The pre-rebase squash is preserved as
tag `netdiff-validated`; `git diff netdiff-validated voicechat_v0.24.0` is **empty** (trees identical).
NOTE: intermediate commits do not individually build (inherent to preserving WIP history); the branch
TIP is the validated, import/functionally-checked tree.

### Reference: net-diff derivation (how the resolved tree was originally produced)

The 146 commits on `voicechat_v0.22.0` are messy WIP history; the resolved tree was first produced by
porting the *net* custom diff onto a fresh upstream base, then replayed as above:

1. `git fetch upstream --tags` (v0.24.0 = `ee0da84ab9e04ac7610e28580af62c365e898389`).
2. Branch `voicechat_v0.24.0` from `v0.24.0`; apply `git diff v0.22.0..voicechat_v0.22.0`
   with `git apply --3way`.
3. **Measured conflict scope (2026-07):** 45 / 55 files apply cleanly (all new model files —
   `eartts.py`, `fastconformer*.py`, `optimized_t5gemma.py`, configs, kernels — add clean).
   **10 files need manual resolution**, all deep integration points that churned in 0.23/0.24:
   - `vllm/config/model.py`
   - `vllm/sampling_params.py`
   - `vllm/transformers_utils/config.py`, `.../configs/__init__.py`
   - `vllm/v1/attention/backend.py`
   - `vllm/v1/core/block_pool.py`
   - `vllm/v1/core/sched/scheduler.py`
   - `vllm/v1/core/single_type_kv_cache_manager.py`
   - `vllm/v1/kv_cache_interface.py`
   - `vllm/v1/worker/gpu_model_runner.py`
   Re-integrate the fork's custom-inputs/outputs hooks, CFG scheduling, await-input scheduling,
   and FastConformer conv-layer KV grouping against the new 0.24.0 code in these files.
4. Rebuild against the `v0.24.0` precompiled wheel, push to `virajkarandikar/vllm`.
5. Update riva-speech `docker/Dockerfile.s2s`: `ARG VLLM_VERSION=0.24.0`, the
   `VLLM_PRECOMPILED_WHEEL` URL → `v0.24.0`, verify `VLLM_CUDA_REF=cu129` wheel exists,
   `git checkout <new commit>`, and re-evaluate the `VLLM_USE_DEEP_GEMM=0` workaround.
6. **Validate end-to-end** (requires GPU build + nemotron-voicechat model) with the S2S
   server startup + inference smoke test before landing.

Until this is done and validated, the 3 vLLM CVEs are handled by temporary GlobalVEX
(not inference-reachable unless spec-decode / M-RoPE prompt_embeds / guided-regex are exposed).

### Conflict-resolution status (branch `voicechat_v0.24.0`, WIP — py-compiles, NOT runtime-validated)

Steps 1–2 above are DONE on the local `voicechat_v0.24.0` branch. **All 10 conflicts are
resolved.** Validation performed (2026-07, inside the riva-s2s A100 container):
- `python -m py_compile` passes on all 10 touched files.
- **Import test PASSES**: `import vllm` + all ported modules import cleanly from source —
  `v1.kv_cache_interface`, `v1.core.block_pool`, `v1.core.single_type_kv_cache_manager`
  (registry rewrite), `v1.core.sched.scheduler` (CFG merge), `v1.attention.backend`,
  `v1.worker.gpu_model_runner`, `config.model`, `sampling_params`, and the
  `model_executor.models.eartts` / `fastconformer` model files. No missing-symbol or broken-import
  errors in any merge.

- **Install PASSES**: with the old-image PEP 639 `license` line patched to `{text=...}` (an
  old-setuptools artifact, not a code issue — a real 0.24.0 build uses current setuptools), the fork
  installs cleanly against the `v0.24.0+cu129` precompiled wheel → `vllm==0.24.0+precompiled`.
  (Install pulls newer deps: numpy 2.3.5, compressed-tensors 0.17, flashinfer 0.6.12, etc. — the
  real Dockerfile update must reconcile these pins.)
- **Functional check PASSES**: `FastConformerConvSpec(block_size=1, shape=(2,128,64), fp16)`
  constructs (`page_size_bytes=32768`), and `spec.is_uniform_with_collection({...})` returns `True`
  — exercising the exact registry merge (base method → `KVCacheSpecRegistry.get_uniform_type_base_spec`
  → the registered `uniform_type_base_spec=FastConformerConvSpec`). The hardest semantic merge works.

Still to do (needs the maintainer's build+deploy cycle):
- Full nemotron-voicechat inference run on 0.24.0 (model repo may need 0.24.0-compat config tweaks).
- Push `voicechat_v0.24.0` to `virajkarandikar/vllm`, update `docker/Dockerfile.s2s`
  (VLLM_VERSION=0.24.0 / wheel URL / commit / re-eval VLLM_USE_DEEP_GEMM), rebuild + smoke test.

**Resolved (mechanical, verified safe):**
- `transformers_utils/configs/__init__.py`, `transformers_utils/config.py` — re-added the
  `FastConformerCTCConfig` / `EarTTSConfig` / `JAISConfig` registry entries (v0.24.0 has none of
  them elsewhere; confirmed no duplicate keys).
- `sampling_params.py` — merged typing imports → `from typing import Annotated, Any, Optional`.
- `config/model.py` — kept v0.24.0 body + fork `custom_input_specs`/`custom_outputs` parsing
  (v0.24.0 has neither the GGUF multimodal check nor custom specs elsewhere).
- `v1/worker/gpu_model_runner.py` — kept v0.24.0 `Sampler(use_fp64_gumbel=...)` **and** fork
  `num_output_tokens_per_step`.

**Resolved (semantic — verified against v0.24.0 APIs; need runtime validation):**
- `v1/core/single_type_kv_cache_manager.py` — v0.24.0 replaced the static `spec_manager_map`
  dict with `KVCacheSpecRegistry` / `get_manager_for_kv_cache_spec`. Kept the fork's
  `FastConformerConvManager`, DROPPED the dead dict, and registered `FastConformerConvSpec` in
  `register_all_kvcache_specs` via `KVCacheSpecRegistry.register(FastConformerConvSpec,
  FastConformerConvManager, uniform_type_base_spec=FastConformerConvSpec)`. Renamed the override
  param `use_eagle` → `drop_eagle_block` to match v0.24.0's base `find_longest_cache_hit`.
- `v1/kv_cache_interface.py` — took v0.24.0's `is_uniform_with_collection` one-liner. FastConformer
  uniformity now flows through the registered `uniform_type_base_spec`: the base method (L143)
  calls `KVCacheSpecRegistry.get_uniform_type_base_spec(self)`, reproducing the fork's
  `isinstance(spec, FastConformerConvSpec)` semantics. (`FastConformerConvSpec` inherits the base
  method — no override needed.)
- `v1/core/block_pool.py` — merged v0.24.0's hash/no-hash eviction ordering (`prepend_n`/`append_n`)
  with the fork's KV-block zeroing tracking (`_freed_block_ids`, init L199, drained L738).
- `v1/attention/backend.py` — merged v0.24.0's `causal` tensor-slicing with the fork's
  `doc_ids` / `decode_offset` fields (class attrs at L424/L427).
- `v1/core/sched/scheduler.py` — merged v0.24.0's `_inflight_prefills` tracking with the fork's CFG
  uncond-pair scheduling + `num_cached_tokens` init (`cfg_uncond` defined L845,
  `scheduled_new_cfg_pairs` L422).

---
## CVEs Requiring vLLM 0.22.0+ (prior 0.19→0.22 migration — DONE, superseded)

> Note: the "Current State" below is stale — the Dockerfile now uses `voicechat_v0.22.0`
> @ `2c089ab05` with the `v0.22.0+cu129` wheel. Kept for history.


| CVE / GHSA | Severity | Package | Fixed In |
|------------|----------|---------|----------|
| GHSA-94f4-hr76-p5j6 | CRITICAL | vllm | 0.22.0 |
| GHSA-q8gq-377p-jq3r | HIGH | vllm | 0.22.0 |
| GHSA-fgcw-684q-jj6r | HIGH | transformers | 5.5.0 |
| GHSA-29pf-2h5f-8g72 | HIGH | transformers | 5.3.0 |

The transformers CVEs are also blocked on the vLLM fork update because
transformers 5.x changed `layer_types` validation in a way that breaks
`vllm/transformers_utils/configs/eartts.py` on the current fork branch.

## Current State (riva-speech Dockerfile.s2s)

- Fork branch: `voicechat_v0.19.0`
- Fork commit: `c044bd6fca81b18fa0f31e6bb839ca612ea5de42`
- Precompiled wheel: `vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl`
- CUDA ref: `cu130`

## What's Needed

1. **Create `voicechat_v0.22.0` branch** in this fork based on upstream vLLM 0.22.0,
   porting the following custom patches from `voicechat_v0.19.0`:
   - KV cache reset fix (`fix-kv-cache-reset` PR / commits `3c0cfcc1`, `ead53a13`)
   - Streaming input fix (`2f3e7baa6`)
   - eartts imports fix (`538e37d42`)
   - Missing imports / renamed variables fixes (`ce6fd2b86`, `4c1c6aabe`, `c80edfbbb`)
   - FastConformer chunk support (`0c5d4d740`)

2. **Verify precompiled wheel** exists for `vllm-0.22.0+cu130`:
   https://github.com/vllm-project/vllm/releases/tag/v0.22.0

3. **Update `docker/Dockerfile.s2s`** in riva-speech:
   ```dockerfile
   ARG VLLM_VERSION=0.22.0
   ...
   git checkout <new-commit-hash-on-voicechat_v0.22.0>
   ```

4. **Re-enable transformers upgrade** after validating 5.x compatibility:
   ```dockerfile
   RUN uv pip install --no-cache --system "transformers>=5.3.0" xgrammar>=0.1.32
   ```
