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

    Grid: (max(1, num_pairs), cdiv(vocab_size, BLOCK_VOCAB))

    Args:
        logits_ptr: Pointer to logits tensor of shape (num_reqs, vocab_size)
        cond_indices_ptr: Pointer to tensor of conditional request indices
        uncond_indices_ptr: Pointer to tensor of unconditional request indices
        scales_ptr: Pointer to tensor of guidance scales per pair
        num_pairs: Number of valid CFG pairs to process
        vocab_size: Vocabulary size
        stride_batch: Stride along batch dimension
        stride_vocab: Stride along vocab dimension
        BLOCK_VOCAB: Block size for vocabulary dimension
    """
    pid_pair = tl.program_id(0)
    pid_vocab = tl.program_id(1)

    # Early exit for padding pairs (handles num_pairs=0 case during CUDA graph capture)
    if pid_pair >= num_pairs:
        return

    # Load indices and scale for this pair
    cond_idx = tl.load(cond_indices_ptr + pid_pair)
    uncond_idx = tl.load(uncond_indices_ptr + pid_pair)
    scale = tl.load(scales_ptr + pid_pair)

    # Compute vocab offsets for this block
    vocab_offsets = pid_vocab * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    vocab_mask = vocab_offsets < vocab_size

    # Compute pointers to logits
    cond_ptrs = logits_ptr + cond_idx * stride_batch + vocab_offsets * stride_vocab
    uncond_ptrs = logits_ptr + uncond_idx * stride_batch + vocab_offsets * stride_vocab

    # Load logits and cast to float32 for numerical stability
    cond_logits = tl.load(cond_ptrs, mask=vocab_mask, other=0.0).to(tl.float32)
    uncond_logits = tl.load(uncond_ptrs, mask=vocab_mask, other=0.0).to(tl.float32)

    # Apply CFG: x = x_cond + scale * (x_cond - x_uncond)
    result = cond_logits + scale * (cond_logits - uncond_logits)

    # Store back to conditional position (in-place, auto-converts to target dtype)
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

    This modifies the conditional logits in-place, combining them with the
    unconditional logits using the guidance scale. After this operation,
    only the conditional logits should be used for sampling.

    Designed for CUDA graph compatibility:
    - Grid uses max(1, num_cfg_pairs) to ensure kernel is always launched
    - When num_cfg_pairs=0, one block is launched but exits immediately
    - Full parallelism preserved: each block handles one pair

    Args:
        logits: Tensor of shape (seq_len, vocab_size) containing packed logits.
                This is a flattened tensor where prefill and generation tokens
                are concatenated. Modified in-place at conditional positions.
        cond_logits_indices: Pre-allocated tensor of shape (max_num_reqs,)
                            containing the token indices (in packed logits)
                            for conditional requests' sampling positions.
                            Only first `num_cfg_pairs` entries are valid.
        uncond_logits_indices: Pre-allocated tensor of shape (max_num_reqs,)
                              containing the token indices (in packed logits)
                              for unconditional requests' sampling positions.
                              Only first `num_cfg_pairs` entries are valid.
        guidance_scales: Pre-allocated tensor of shape (max_num_reqs,) containing
                        guidance scale for each CFG pair.
                        Only first `num_cfg_pairs` entries are valid.
        num_cfg_pairs: Number of valid CFG pairs to process.
    """
    vocab_size = logits.shape[1]

    BLOCK_VOCAB = 1024

    # Use max(1, num_cfg_pairs) to ensure kernel is always launched for CUDA graphs
    # When num_cfg_pairs=0, the single block will exit immediately via bounds check
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

    For each token where mask[seq_idx] == True, the entire embedding
    row is set to the null_emb values.

    Grid: (max(1, cdiv(num_tokens, BLOCK_SEQ)), cdiv(dim, BLOCK_DIM))

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

    # Mask for valid positions (handles num_tokens=0 case during CUDA graph capture)
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


@triton.jit
def _copy_columns_by_indices_kernel(
    data_ptr,
    src_indices_ptr,
    dst_indices_ptr,
    num_pairs,
    num_rows,
    stride_row,
    stride_col,
    BLOCK_ROWS: tl.constexpr,
):
    """
    Copy columns in a 2D tensor from source positions to destination positions.

    For each valid pair i (i < num_pairs):
        data[:, dst_indices[i]] = data[:, src_indices[i]]

    Grid: (max(1, num_pairs), cdiv(num_rows, BLOCK_ROWS))

    Args:
        data_ptr: Pointer to 2D tensor of shape (num_rows, num_cols)
        src_indices_ptr: Pointer to tensor of source column indices
        dst_indices_ptr: Pointer to tensor of destination column indices
        num_pairs: Number of valid pairs to process
        num_rows: Number of rows in the data tensor
        stride_row: Stride along the row dimension
        stride_col: Stride along the column dimension
        BLOCK_ROWS: Block size for the row dimension
    """
    pid_pair = tl.program_id(0)
    pid_row = tl.program_id(1)

    # Early exit for padding pairs (handles num_pairs=0 for CUDA graph capture)
    if pid_pair >= num_pairs:
        return

    # Load source and destination column indices for this pair
    src_col = tl.load(src_indices_ptr + pid_pair)
    dst_col = tl.load(dst_indices_ptr + pid_pair)

    # Compute row offsets for this block
    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offsets < num_rows

    # Compute pointers
    src_ptrs = data_ptr + row_offsets * stride_row + src_col * stride_col
    dst_ptrs = data_ptr + row_offsets * stride_row + dst_col * stride_col

    # Load from source and store to destination
    values = tl.load(src_ptrs, mask=row_mask)
    tl.store(dst_ptrs, values, mask=row_mask)


def copy_columns_by_indices(
    data: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    num_pairs: int,
) -> None:
    """
    Copy columns in a 2D tensor from source positions to destination
    positions, in-place.

    For each valid pair i (i < num_pairs):
        data[:, dst_indices[i]] = data[:, src_indices[i]]

    Designed for CUDA graph compatibility:
    - Grid uses max(1, num_pairs) to ensure kernel is always launched
    - When num_pairs=0, one block is launched but exits immediately
    - Pre-allocated index tensors can be used at full size; only
      the first ``num_pairs`` entries are read

    Args:
        data: Tensor of shape (num_rows, num_cols). Modified in-place.
        src_indices: Pre-allocated tensor of shape (max_num_pairs,)
                     containing source column indices.
                     Only first ``num_pairs`` entries are used.
        dst_indices: Pre-allocated tensor of shape (max_num_pairs,)
                     containing destination column indices.
                     Only first ``num_pairs`` entries are used.
        num_pairs: Number of valid pairs to process.
    """
    num_rows = data.shape[0]

    BLOCK_ROWS = 32

    grid = (
        max(1, num_pairs),
        triton.cdiv(num_rows, BLOCK_ROWS),
    )

    _copy_columns_by_indices_kernel[grid](
        data,
        src_indices,
        dst_indices,
        num_pairs,
        num_rows,
        data.stride(0),
        data.stride(1),
        BLOCK_ROWS=BLOCK_ROWS,
    )


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

    Designed for CUDA graph compatibility:
    - Grid uses max(1, cdiv(num_tokens, BLOCK_SEQ)) to ensure kernel is always launched
    - When num_tokens=0, one block is launched but seq_mask is all False (no stores)
    - Full parallelism preserved: each block handles a chunk of tokens

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
    dim = embeddings.shape[1]

    BLOCK_SEQ = 32
    BLOCK_DIM = 128

    # Use max(1, ...) to ensure kernel is always launched for CUDA graphs
    # When num_tokens=0, the single seq block will have seq_mask all False (no stores)
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
