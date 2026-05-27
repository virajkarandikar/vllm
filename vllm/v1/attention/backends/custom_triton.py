# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Triton attention backend.

This backend uses a custom unified attention wrapper that can override tile
sizes via env vars while reusing the underlying Triton kernels.
"""

import os
import re
from typing import Callable

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.custom_unified_attention import custom_unified_attention
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

# Observer: (layer_idx, head_idx, attn, batch_idx=None). batch_idx is provided for
# batched decode so the analyzer can maintain per-request state.
_alignment_observer: Callable[..., None] | None = None

# Keep deterministic order to match analyzer indexing.
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]

# ── Fused alignment path (CHATTERBOX_FUSED_ALIGNMENT=1) ──
# Per-layer batched attention results stored on GPU for T3 to consume.
_fused_attn_buffers: dict[tuple[int, int], torch.Tensor] = {}
_fused_decode_meta: dict[str, object] = {}

FUSED_ALIGNMENT = os.environ.get("CHATTERBOX_FUSED_ALIGNMENT", "1") == "1"


def get_fused_attn_buffers() -> dict[tuple[int, int], torch.Tensor]:
    return _fused_attn_buffers


def get_fused_decode_meta() -> dict[str, object]:
    return _fused_decode_meta


def clear_fused_attn_buffers() -> None:
    _fused_attn_buffers.clear()


def register_alignment_observer(
    observer: Callable[..., None] | None,
) -> None:
    """Register a callback for alignment attention. Signature: (layer_idx, head_idx, attn, batch_idx=None)."""
    global _alignment_observer
    _alignment_observer = observer


def _extract_layer_index(layer: torch.nn.Module) -> int | None:
    name = getattr(layer, "layer_name", "")
    if not name:
        return None
    match = re.search(r"layers\.(\d+)\.", name)
    if not match:
        return None
    return int(match.group(1))


def _compute_alignment_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    head_idx: int,
    num_queries_per_kv: int,
    scale: float,
) -> torch.Tensor | None:
    """Single-request path: only when batch size is 1."""
    if seq_lens.numel() != 1 or block_table.shape[0] != 1:
        return None
    return _compute_alignment_attention_for_request(
        request_idx=0,
        query=query,
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=None,
        head_idx=head_idx,
        num_queries_per_kv=num_queries_per_kv,
        scale=scale,
    )


def _compute_alignment_attention_for_request(
    request_idx: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    head_idx: int,
    num_queries_per_kv: int,
    scale: float,
) -> torch.Tensor | None:
    """Compute alignment attention for one request (decode: one query token)."""
    num_seqs = seq_lens.numel()
    if request_idx < 0 or request_idx >= num_seqs:
        return None
    seq_len = int(seq_lens[request_idx].item())
    if seq_len == 0:
        return None

    block_size = key_cache.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    block_ids = block_table[request_idx, :num_blocks].to(dtype=torch.int64)

    k_blocks = key_cache.index_select(0, block_ids)
    k_seq = k_blocks.reshape(-1, key_cache.shape[2], key_cache.shape[3])[:seq_len]

    kv_head_idx = head_idx // num_queries_per_kv
    # Decode: one query token per request. Packed layout: query row for request b is at cu_seqlens_q[b+1]-1.
    if cu_seqlens_q is not None:
        q_token_idx = int(cu_seqlens_q[request_idx + 1].item()) - 1
        q = query[q_token_idx, head_idx]
    else:
        q = query[-1, head_idx]
    k = k_seq[:, kv_head_idx]

    attn_scores = (q @ k.T) * scale
    attn = torch.softmax(attn_scores, dim=-1)
    return attn.unsqueeze(0).detach().cpu()


@register_backend(AttentionBackendEnum.CUSTOM)
class CustomTritonAttentionBackend(TritonAttentionBackend):
    """Custom backend that delegates to Triton attention implementation."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TritonAttentionImpl]:
        return CustomTritonAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        return TritonAttentionMetadataBuilder


class CustomTritonAttentionImpl(TritonAttentionImpl):
    """Custom attention impl that calls the custom unified attention wrapper."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for CustomTritonAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        custom_unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        self._maybe_emit_alignment(
            layer=layer,
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
        )

        return output

    def _maybe_emit_alignment(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
    ) -> None:
        if os.environ.get("CHATTERBOX_ENABLE_ALIGNMENT_ANALYZER", "0") != "1":
            return
        if max_seqlen_q != 1:
            return

        layer_idx = _extract_layer_index(layer)
        if layer_idx is None:
            return

        if FUSED_ALIGNMENT:
            self._fused_emit_alignment(
                layer_idx, query, key_cache, block_table, seq_lens, cu_seqlens_q,
            )
        else:
            self._legacy_emit_alignment(
                layer_idx, query, key_cache, block_table, seq_lens, cu_seqlens_q,
            )

    def _fused_emit_alignment(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        """Batched GPU extraction — fully vectorized, zero Python loops."""
        num_seqs = seq_lens.numel()
        if num_seqs == 0:
            return
        block_size = key_cache.shape[1]
        head_size = key_cache.shape[3]
        device = query.device
        max_sl = seq_lens.max()  # keep on GPU

        # Pre-compute q indices for all requests (decode: last token per seq)
        q_indices = cu_seqlens_q[1:num_seqs + 1] - 1  # [num_seqs] on GPU

        # Build flat K indices: for each request, gather block_table entries
        # and expand to per-position indices. Use a mask for variable lengths.
        max_blocks = block_table.shape[1]
        # positions within the flattened block layout: [num_seqs, max_blocks * block_size]
        max_positions = max_blocks * block_size
        # Block IDs per request: [num_seqs, max_blocks]
        blk_ids = block_table[:num_seqs].to(torch.int64)

        for layer_cand, head_cand in LLAMA_ALIGNED_HEADS:
            if layer_cand != layer_idx:
                continue
            kv_head = head_cand // self.num_queries_per_kv

            # Gather Q vectors: [num_seqs, head_size]
            q_vecs = query[q_indices, head_cand]  # advanced indexing, stays on GPU

            # Gather K for all requests in one shot using block_table
            # key_cache: [num_blks, blk_size, num_kv_heads, head_size]
            # blk_ids:   [num_seqs, max_blocks]
            k_blocks = key_cache[blk_ids, :, kv_head, :]  # [num_seqs, max_blocks, blk_size, head_size]
            k_flat = k_blocks.reshape(num_seqs, -1, head_size)  # [num_seqs, max_positions, head_size]

            # Batched matmul: Q [num_seqs, 1, head_size] @ K^T [num_seqs, head_size, max_positions]
            # Cast both to float32 for numerical stability (Q/K may be bf16)
            scores = torch.bmm(
                q_vecs.unsqueeze(1).float(),    # [num_seqs, 1, head_size]
                k_flat.transpose(1, 2).float(), # [num_seqs, head_size, max_positions]
            ).squeeze(1) * self.scale            # [num_seqs, max_positions]

            # Mask out positions beyond each request's seq_len
            pos_idx = torch.arange(scores.shape[1], device=device).unsqueeze(0)  # [1, max_positions]
            mask = pos_idx >= seq_lens[:num_seqs].unsqueeze(1)  # [num_seqs, max_positions]
            scores.masked_fill_(mask, float('-inf'))

            # Softmax over valid positions
            attn_batch = torch.softmax(scores, dim=-1)  # [num_seqs, max_positions]
            attn_batch.masked_fill_(mask, 0.0)

            _fused_attn_buffers[(layer_idx, head_cand)] = attn_batch

        _fused_decode_meta["num_seqs"] = num_seqs
        _fused_decode_meta["seq_lens"] = seq_lens
        _fused_decode_meta["num_queries_per_kv"] = self.num_queries_per_kv
        _fused_decode_meta["scale"] = self.scale

    def _legacy_emit_alignment(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        """Original per-request CPU path (observer callback)."""
        if _alignment_observer is None:
            return
        num_seqs = seq_lens.numel()
        for layer_cand, head_cand in LLAMA_ALIGNED_HEADS:
            if layer_cand != layer_idx:
                continue
            for batch_idx in range(num_seqs):
                attn = _compute_alignment_attention_for_request(
                    request_idx=batch_idx,
                    query=query,
                    key_cache=key_cache,
                    block_table=block_table,
                    seq_lens=seq_lens,
                    cu_seqlens_q=cu_seqlens_q,
                    head_idx=head_cand,
                    num_queries_per_kv=self.num_queries_per_kv,
                    scale=self.scale,
                )
                if attn is not None:
                    if num_seqs == 1:
                        _alignment_observer(layer_idx, head_cand, attn)
                    else:
                        try:
                            _alignment_observer(layer_idx, head_cand, attn, batch_idx)
                        except TypeError:
                            pass

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            sm_scale=self.scale,
            causal=False,
        )
        return output
