# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernels for Classifier Free Guidance (CFG) operations.

These kernels are optimized for setting unconditional prefill embeddings
to a null embedding value, which is a key step in CFG inference.
Designed for CUDA graph compatibility with pre-allocated tensors.
"""

import torch

from vllm.triton_utils import tl, triton


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

    For each token where mask[seq_idx] == True, the entire embedding
    row is set to the null_emb values.

    Args:
        embeddings_ptr: Pointer to embeddings tensor of shape (seq_len, dim)
        null_emb_ptr: Pointer to null embedding tensor of shape (dim,)
        mask_ptr: Pointer to boolean mask tensor of shape (max_num_tokens,)
                  True indicates unconditional tokens to be set to null_emb
        num_tokens: Number of valid tokens to process
        dim: Embedding dimension
        stride_seq: Stride along sequence dimension
        stride_dim: Stride along embedding dimension
        BLOCK_SEQ: Block size for sequence dimension
        BLOCK_DIM: Block size for embedding dimension
    """
    # Each program handles a block of sequences and a block of dimensions
    pid_seq = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Compute offsets for this block
    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    # Mask for valid positions
    seq_mask = seq_offsets < num_tokens
    dim_mask = dim_offsets < dim

    # Load the uncond mask for this block of sequences
    uncond_mask = tl.load(
        mask_ptr + seq_offsets,
        mask=seq_mask,
        other=False,
    )

    # Load the null embedding values for this block of dimensions
    null_emb_values = tl.load(
        null_emb_ptr + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )

    # Compute pointers to embedding elements
    emb_ptrs = (
        embeddings_ptr
        + seq_offsets[:, None] * stride_seq
        + dim_offsets[None, :] * stride_dim
    )

    # Combined mask: valid positions AND uncond tokens
    combined_mask = seq_mask[:, None] & dim_mask[None, :] & uncond_mask[:, None]

    # Broadcast null_emb to (BLOCK_SEQ, BLOCK_DIM) and write in-place
    # null_emb_values is (BLOCK_DIM,), broadcast to all sequence positions
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

    This function is used during CFG (Classifier Free Guidance) prefill
    to set embeddings for unconditional requests to a learned null embedding,
    which represents the unconditional path.

    Designed for CUDA graph compatibility - all tensors should be pre-allocated.

    Args:
        embeddings: Tensor of shape (seq_len, dim) containing token embeddings.
                   Modified in-place.
        null_emb: Tensor of shape (dim,) containing the null embedding values
                 to set for unconditional tokens.
        uncond_token_mask: Boolean tensor of shape (max_num_tokens,) where True
                          indicates tokens that belong to unconditional requests
                          and should be set to null_emb. Pre-allocated for
                          CUDA graphs.
        num_tokens: Number of valid tokens to process.
    """
    if num_tokens == 0:
        return

    dim = embeddings.shape[1]

    BLOCK_SEQ = 32
    BLOCK_DIM = 128

    grid = (
        triton.cdiv(num_tokens, BLOCK_SEQ),
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
