# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernels for Classifier Free Guidance (CFG) operations.

These kernels are optimized for:
1. Setting unconditional prefill embeddings to a null embedding value
2. Applying CFG formula to logits: x = x_cond + scale * (x_cond - x_uncond)

Designed for CUDA graph compatibility with pre-allocated tensors.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _apply_cfg_logits_kernel(
    logits_ptr,
    cond_indices_ptr,
    uncond_indices_ptr,
    scales_ptr,
    num_pairs,
    vocab_size,
    stride_batch,
    stride_vocab,
    BLOCK_VOCAB: tl.constexpr,
):
    """
    Apply CFG formula in-place to conditional logits.

    For each CFG pair:
        logits[cond_idx] = logits[cond_idx] + scale * (logits[cond_idx] - logits[uncond_idx])
    """
    pid_pair = tl.program_id(0)
    pid_vocab = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    cond_idx = tl.load(cond_indices_ptr + pid_pair)
    uncond_idx = tl.load(uncond_indices_ptr + pid_pair)
    scale = tl.load(scales_ptr + pid_pair)

    vocab_offsets = pid_vocab * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    vocab_mask = vocab_offsets < vocab_size

    cond_ptrs = logits_ptr + cond_idx * stride_batch + vocab_offsets * stride_vocab
    uncond_ptrs = logits_ptr + uncond_idx * stride_batch + vocab_offsets * stride_vocab

    cond_logits = tl.load(cond_ptrs, mask=vocab_mask, other=0.0).to(tl.float32)
    uncond_logits = tl.load(uncond_ptrs, mask=vocab_mask, other=0.0).to(tl.float32)

    result = cond_logits + scale * (cond_logits - uncond_logits)

    tl.store(cond_ptrs, result, mask=vocab_mask)


def apply_cfg_logits(
    logits: torch.Tensor,
    cond_logits_indices: torch.Tensor,
    uncond_logits_indices: torch.Tensor,
    guidance_scales: torch.Tensor,
    num_cfg_pairs: int,
) -> None:
    """
    Apply CFG formula to logits in-place for decode/sampling positions.

    Formula: logits[cond] = logits[cond] + scale * (logits[cond] - logits[uncond])
    """
    vocab_size = logits.shape[1]

    BLOCK_VOCAB = 1024

    grid = (
        max(1, num_cfg_pairs),
        triton.cdiv(vocab_size, BLOCK_VOCAB),
    )

    _apply_cfg_logits_kernel[grid](
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs,
        vocab_size,
        logits.stride(0),
        logits.stride(1),
        BLOCK_VOCAB=BLOCK_VOCAB,
    )


@triton.jit
def _set_uncond_embeddings_kernel(
    embeddings_ptr,
    null_emb_ptr,
    mask_ptr,
    num_tokens,
    dim,
    stride_seq,
    stride_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Triton kernel to set embeddings for unconditional tokens to null_emb
    in-place.
    """
    pid_seq = tl.program_id(0)
    pid_dim = tl.program_id(1)

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    seq_mask = seq_offsets < num_tokens
    dim_mask = dim_offsets < dim

    uncond_mask = tl.load(
        mask_ptr + seq_offsets,
        mask=seq_mask,
        other=False,
    )

    null_emb_values = tl.load(
        null_emb_ptr + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )

    emb_ptrs = (
        embeddings_ptr
        + seq_offsets[:, None] * stride_seq
        + dim_offsets[None, :] * stride_dim
    )

    combined_mask = seq_mask[:, None] & dim_mask[None, :] & uncond_mask[:, None]

    null_emb_broadcast = tl.broadcast_to(
        null_emb_values[None, :], (BLOCK_SEQ, BLOCK_DIM)
    )
    tl.store(emb_ptrs, null_emb_broadcast, mask=combined_mask)


def set_uncond_embeddings(
    embeddings: torch.Tensor,
    null_emb: torch.Tensor,
    uncond_token_mask: torch.Tensor,
    num_tokens: int,
) -> None:
    """
    Set embeddings for unconditional tokens to null_emb values in-place.
    """
    dim = embeddings.shape[1]

    BLOCK_SEQ = 32
    BLOCK_DIM = 128

    grid = (
        max(1, triton.cdiv(num_tokens, BLOCK_SEQ)),
        triton.cdiv(dim, BLOCK_DIM),
    )

    _set_uncond_embeddings_kernel[grid](
        embeddings,
        null_emb,
        uncond_token_mask,
        num_tokens,
        dim,
        embeddings.stride(0),
        embeddings.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
    )
