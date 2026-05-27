# Custom vLLM Changes

This document describes the custom modifications made to the vLLM codebase for benchmarking and Chatterbox TTS alignment analysis.

## Overview

The following custom files and changes have been added to enable:
1. **Custom tile size benchmarking** - Override Triton attention tile sizes via environment variables
2. **Alignment analysis** - Extract attention weights from specific Llama heads for TTS alignment
3. **Pluggable custom backends** - Register custom attention backends without modifying core vLLM

---

## New Files

### 1. `vllm/v1/attention/backends/custom_triton.py`

A custom Triton attention backend that extends the base `TritonAttentionBackend`.

**Key Features:**
- Registered as `AttentionBackendEnum.CUSTOM` using the `@register_backend` decorator
- Calls `custom_unified_attention` wrapper for tile size overrides
- Supports alignment extraction for Chatterbox TTS

**Alignment Analysis:**
- Predefined attention heads for alignment: `LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]`
- `register_alignment_observer(callback)` - Attach an external observer to receive attention weights
- `_compute_alignment_attention()` - Extracts softmax attention from specific layer/head combinations

**Usage:**
```python
from vllm.v1.attention.backends.custom_triton import register_alignment_observer

def my_observer(layer_idx: int, head_idx: int, attn: torch.Tensor):
    print(f"Layer {layer_idx}, Head {head_idx}: {attn.shape}")

register_alignment_observer(my_observer)
```

---

### 2. `vllm/v1/attention/ops/custom_unified_attention.py`

A wrapper around the base Triton unified attention that allows tile size overrides.

**Key Features:**
- `_get_tile_size_override()` - Checks for environment variable overrides before using defaults
- `custom_unified_attention()` - Main entry point that uses overridable tile sizes

**How it works:**
1. Checks for `VLLM_CUSTOM_TILE_SIZE_PREFILL` or `VLLM_CUSTOM_TILE_SIZE_DECODE` env vars
2. If set to a positive integer, uses that value as the tile size
3. Otherwise, falls back to the base `_get_tile_size()` function

---

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_CUSTOM_TILE_SIZE_PREFILL` | int | (auto) | Override tile size for prefill operations |
| `VLLM_CUSTOM_TILE_SIZE_DECODE` | int | (auto) | Override tile size for decode operations |
| `CHATTERBOX_ENABLE_ALIGNMENT_ANALYZER` | "0"/"1" | "0" | Enable alignment extraction from attention heads |
| `CHATTERBOX_ALIGNMENT_DEBUG` | "0"/"1" | "0" | Enable debug logging for alignment extraction |

---

## Usage Examples

### Benchmarking with Custom Tile Sizes

```bash
# Set custom tile sizes for benchmarking
export VLLM_CUSTOM_TILE_SIZE_PREFILL=64
export VLLM_CUSTOM_TILE_SIZE_DECODE=32

# Run vLLM with custom attention backend
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --attention-backend CUSTOM
```

### Enabling Alignment Analysis for Chatterbox TTS

```bash
# Enable alignment extraction
export CHATTERBOX_ENABLE_ALIGNMENT_ANALYZER=1
export CHATTERBOX_ALIGNMENT_DEBUG=1  # Optional: for debug output

# Run your Chatterbox TTS inference
python your_tts_script.py
```

### Programmatic Alignment Observer

```python
import torch
from vllm.v1.attention.backends.custom_triton import register_alignment_observer

# Store alignment data
alignment_data = []

def alignment_callback(layer_idx: int, head_idx: int, attn: torch.Tensor):
    """Called during decode steps for specified layer/head combinations."""
    alignment_data.append({
        "layer": layer_idx,
        "head": head_idx,
        "attention": attn.clone()
    })

# Register before running inference
register_alignment_observer(alignment_callback)

# ... run inference ...

# Process collected alignments
for item in alignment_data:
    print(f"Layer {item['layer']}, Head {item['head']}: {item['attention'].shape}")
```

---

## Aligned Heads Configuration

The default aligned heads for Llama models are defined in `custom_triton.py`:

```python
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]
```

Each tuple is `(layer_index, head_index)`. These were selected based on analysis of attention patterns that correlate with text-audio alignment in TTS applications.

---

## Architecture

```
vllm/v1/attention/
├── backends/
│   ├── custom_triton.py      # Custom backend with alignment support
│   ├── registry.py           # Backend registration (CUSTOM enum added)
│   └── ...
└── ops/
    ├── custom_unified_attention.py  # Tile size override wrapper
    ├── triton_unified_attention.py  # Base unified attention (unchanged)
    └── ...
```

---

## Notes

- The custom backend only supports **causal attention** (decoder-only models)
- Alignment extraction only occurs during **decode steps** (`max_seqlen_q == 1`)
- Alignment extraction requires **single-sequence batches** for accurate results
- FP8 quantization is supported but Q-scale must be 1.0
