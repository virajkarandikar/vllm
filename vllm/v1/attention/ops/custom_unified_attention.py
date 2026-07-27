# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom unified attention wrapper with overridable tile sizes."""

from __future__ import annotations

import contextlib
import os

import torch

from vllm.v1.attention.ops import triton_unified_attention as base_unified


def _env_tile_size(name: str) -> int | None:
    """Read a positive int tile size from ``name``, or None if unset/invalid."""
    override = os.getenv(name)
    if not override:
        return None
    try:
        value = int(override)
    except ValueError:
        return None
    return value if value > 0 else None


@contextlib.contextmanager
def _tile_size_override():
    """Scope an env override of the base tile-size heuristic.

    ``unified_attention`` picks its tile sizes internally via
    ``_get_tile_size``; there is no parameter to inject them.  Rather than
    fork the whole wrapper (~80 kernel kwargs that would have to be kept in
    sync with upstream by hand), swap the module-level helper for the
    duration of the call.

    When neither env var is set this is a no-op and the delegated call is
    bit-identical to upstream.  vLLM runs the forward pass single-threaded
    per worker process, so the temporary rebind is not raced.
    """
    prefill = _env_tile_size("VLLM_CUSTOM_TILE_SIZE_PREFILL")
    decode = _env_tile_size("VLLM_CUSTOM_TILE_SIZE_DECODE")
    if prefill is None and decode is None:
        yield
        return

    original = base_unified._get_tile_size

    def patched(head_size, sliding_window, element_size, is_prefill):
        override = prefill if is_prefill else decode
        if override is not None:
            return override
        return original(
            head_size=head_size,
            sliding_window=sliding_window,
            element_size=element_size,
            is_prefill=is_prefill,
        )

    base_unified._get_tile_size = patched
    try:
        yield
    finally:
        base_unified._get_tile_size = original


def custom_unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    qq_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    mm_prefix_range: torch.Tensor | None = None,
    use_alibi_sqrt: bool = False,
) -> None:
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    with _tile_size_override():
        base_unified.unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            alibi_slopes=alibi_slopes,
            output_scale=output_scale,
            qq_bias=qq_bias,
            sinks=sinks,
            mm_prefix_range=mm_prefix_range,
            use_alibi_sqrt=use_alibi_sqrt,
        )
