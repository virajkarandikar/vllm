# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CFG (Classifier Free Guidance) operations.

Run `pytest tests/kernels/core/test_cfg_ops.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.cfg_ops import (
    apply_cfg_logits,
    copy_columns_by_indices,
    set_uncond_embeddings,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_tokens", [1, 64, 256])
@pytest.mark.parametrize("hidden_size", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_set_uncond_embeddings_basic(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
):
    """Test basic functionality of set_uncond_embeddings."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    # Create input embeddings
    embeddings = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    original_embeddings = embeddings.clone()

    # Create null embedding
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Create mask where all tokens are unconditional
    uncond_mask = torch.ones(num_tokens, device="cuda", dtype=torch.bool)

    # Apply kernel
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, num_tokens)

    # All rows should be set to null_emb
    expected = null_emb.unsqueeze(0).expand(num_tokens, -1)
    torch.testing.assert_close(embeddings, expected, atol=1e-5, rtol=1e-5)

    # Verify shape and dtype preserved
    assert embeddings.shape == original_embeddings.shape
    assert embeddings.dtype == dtype


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("hidden_size", [256, 512])
def test_set_uncond_embeddings_partial_mask(num_tokens: int, hidden_size: int):
    """Test set_uncond_embeddings with partial mask (some tokens conditional)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    dtype = torch.float16

    # Create input embeddings
    embeddings = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    original_embeddings = embeddings.clone()

    # Create null embedding
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Create mask where only even indices are unconditional
    uncond_mask = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)
    uncond_mask[::2] = True  # Every other token is unconditional

    # Apply kernel
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, num_tokens)

    # Check that unconditional tokens are set to null_emb
    for i in range(num_tokens):
        if i % 2 == 0:
            torch.testing.assert_close(embeddings[i], null_emb, atol=1e-5, rtol=1e-5)
        else:
            # Conditional tokens should be unchanged
            torch.testing.assert_close(
                embeddings[i], original_embeddings[i], atol=0, rtol=0
            )


def test_set_uncond_embeddings_no_uncond():
    """Test set_uncond_embeddings with no unconditional tokens."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 64
    hidden_size = 512
    dtype = torch.float16

    # Create input embeddings
    embeddings = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    original_embeddings = embeddings.clone()

    # Create null embedding
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Create mask where no tokens are unconditional
    uncond_mask = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)

    # Apply kernel
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, num_tokens)

    # All tokens should be unchanged
    torch.testing.assert_close(embeddings, original_embeddings, atol=0, rtol=0)


def test_set_uncond_embeddings_empty():
    """Test set_uncond_embeddings with zero tokens."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    dtype = torch.float16
    hidden_size = 256

    # Create empty embeddings tensor (but valid shape)
    embeddings = torch.randn(0, hidden_size, device="cuda", dtype=dtype)
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)
    uncond_mask = torch.zeros(0, device="cuda", dtype=torch.bool)

    # Should not raise, just return early
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, num_tokens=0)


def test_set_uncond_embeddings_cfg_scenario():
    """Test set_uncond_embeddings in a CFG-like scenario with paired requests."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 64
    hidden_size = 512
    dtype = torch.float16

    # Simulate CFG: half the tokens are conditional, half unconditional
    total_tokens = num_tokens * 2

    # Create input embeddings
    embeddings = torch.randn(total_tokens, hidden_size, device="cuda", dtype=dtype)
    original_embeddings = embeddings.clone()

    # Create null embedding (learned parameter in the model)
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Mask: second half are unconditional
    uncond_mask = torch.zeros(total_tokens, device="cuda", dtype=torch.bool)
    uncond_mask[num_tokens:] = True

    # Apply kernel
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, total_tokens)

    # Conditional tokens (first half) should be unchanged
    torch.testing.assert_close(
        embeddings[:num_tokens], original_embeddings[:num_tokens], atol=0, rtol=0
    )

    # Unconditional tokens (second half) should be set to null_emb
    expected_uncond = null_emb.unsqueeze(0).expand(num_tokens, -1)
    torch.testing.assert_close(
        embeddings[num_tokens:], expected_uncond, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("num_tokens,hidden_size", [(33, 127), (65, 513)])
def test_set_uncond_embeddings_irregular_sizes(num_tokens: int, hidden_size: int):
    """Test set_uncond_embeddings with non-power-of-2 tensor sizes."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    dtype = torch.float16

    # Create input embeddings
    embeddings = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)

    # Create null embedding
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Create random mask
    torch.manual_seed(42)
    uncond_mask = torch.rand(num_tokens, device="cuda") > 0.5

    # Store original for comparison
    original_embeddings = embeddings.clone()

    # Apply kernel
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, num_tokens)

    # Verify results
    for i in range(num_tokens):
        if uncond_mask[i]:
            torch.testing.assert_close(embeddings[i], null_emb, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(
                embeddings[i], original_embeddings[i], atol=0, rtol=0
            )


def test_set_uncond_embeddings_preallocated_mask():
    """Test set_uncond_embeddings with pre-allocated mask (CUDA graph scenario)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    max_num_tokens = 256
    actual_tokens = 64
    hidden_size = 512
    dtype = torch.float16

    # Pre-allocate tensors (as would be done for CUDA graphs)
    embeddings = torch.randn(max_num_tokens, hidden_size, device="cuda", dtype=dtype)
    null_emb = torch.randn(hidden_size, device="cuda", dtype=dtype)
    uncond_mask = torch.zeros(max_num_tokens, device="cuda", dtype=torch.bool)

    # Set up actual data
    original_embeddings = embeddings.clone()
    uncond_mask[:actual_tokens:2] = True  # Every other token in valid range

    # Apply kernel with actual token count
    set_uncond_embeddings(embeddings, null_emb, uncond_mask, actual_tokens)

    # Verify only the first actual_tokens are modified
    for i in range(actual_tokens):
        if i % 2 == 0:
            torch.testing.assert_close(embeddings[i], null_emb, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(
                embeddings[i], original_embeddings[i], atol=0, rtol=0
            )

    # Tokens beyond actual_tokens should be unchanged
    torch.testing.assert_close(
        embeddings[actual_tokens:], original_embeddings[actual_tokens:], atol=0, rtol=0
    )


# ============================================================================
# Tests for apply_cfg_logits kernel
# ============================================================================


@pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_apply_cfg_logits_basic(vocab_size: int, dtype: torch.dtype):
    """Test basic functionality of apply_cfg_logits."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 4
    num_cfg_pairs = 1

    # Create packed logits tensor (seq_len, vocab_size)
    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype)
    original_logits = logits.clone()

    # Token indices: cond at position 0, uncond at position 1
    cond_logits_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    uncond_logits_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    guidance_scales = torch.tensor([3.0], dtype=torch.float32, device="cuda")

    # Apply CFG kernel
    apply_cfg_logits(
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs,
    )

    # Verify CFG formula: x = x_cond + scale * (x_cond - x_uncond)
    # Compute expected in float32, then convert to target dtype (matches kernel behavior)
    expected = original_logits[0].float() + 3.0 * (
        original_logits[0].float() - original_logits[1].float()
    )
    expected = expected.to(dtype)
    torch.testing.assert_close(logits[0], expected, atol=1e-4, rtol=1e-4)

    # Unconditional logits should be unchanged
    torch.testing.assert_close(logits[1], original_logits[1], atol=0, rtol=0)

    # Other tokens should be unchanged
    torch.testing.assert_close(logits[2:], original_logits[2:], atol=0, rtol=0)


def test_apply_cfg_logits_multiple_pairs():
    """Test apply_cfg_logits with multiple CFG pairs."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 6
    vocab_size = 1024
    num_cfg_pairs = 2
    dtype = torch.float16

    # Create packed logits tensor
    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype)
    original_logits = logits.clone()

    # Two CFG pairs:
    # Pair 0: cond at 0, uncond at 1, scale 2.0
    # Pair 1: cond at 3, uncond at 4, scale 5.0
    cond_logits_indices = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    uncond_logits_indices = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    guidance_scales = torch.tensor([2.0, 5.0], dtype=torch.float32, device="cuda")

    apply_cfg_logits(
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs,
    )

    # Verify pair 0 (compute in float32, convert to target dtype)
    expected_0 = (
        original_logits[0].float()
        + 2.0 * (original_logits[0].float() - original_logits[1].float())
    ).to(dtype)
    torch.testing.assert_close(logits[0], expected_0, atol=1e-4, rtol=1e-4)

    # Verify pair 1 (compute in float32, convert to target dtype)
    expected_3 = (
        original_logits[3].float()
        + 5.0 * (original_logits[3].float() - original_logits[4].float())
    ).to(dtype)
    torch.testing.assert_close(logits[3], expected_3, atol=1e-4, rtol=1e-4)

    # Uncond positions should be unchanged
    torch.testing.assert_close(logits[1], original_logits[1], atol=0, rtol=0)
    torch.testing.assert_close(logits[4], original_logits[4], atol=0, rtol=0)

    # Other tokens should be unchanged
    torch.testing.assert_close(logits[2], original_logits[2], atol=0, rtol=0)
    torch.testing.assert_close(logits[5], original_logits[5], atol=0, rtol=0)


def test_apply_cfg_logits_zero_scale():
    """Test apply_cfg_logits with zero guidance scale (no change)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 4
    vocab_size = 2048
    dtype = torch.float16

    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype)
    original_logits = logits.clone()

    cond_logits_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    uncond_logits_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    guidance_scales = torch.tensor([0.0], dtype=torch.float32, device="cuda")

    apply_cfg_logits(
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs=1,
    )

    # With scale=0: x = x_cond + 0 * (x_cond - x_uncond) = x_cond
    torch.testing.assert_close(logits[0], original_logits[0], atol=1e-5, rtol=1e-5)


def test_apply_cfg_logits_no_pairs():
    """Test apply_cfg_logits with zero pairs (should be no-op)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 4
    vocab_size = 1024
    dtype = torch.float16

    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype)
    original_logits = logits.clone()

    # Pre-allocated buffers but zero pairs
    cond_logits_indices = torch.zeros(10, dtype=torch.int32, device="cuda")
    uncond_logits_indices = torch.zeros(10, dtype=torch.int32, device="cuda")
    guidance_scales = torch.ones(10, dtype=torch.float32, device="cuda")

    apply_cfg_logits(
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs=0,
    )

    # No changes should be made
    torch.testing.assert_close(logits, original_logits, atol=0, rtol=0)


def test_apply_cfg_logits_preallocated_buffers():
    """Test apply_cfg_logits with pre-allocated buffers (CUDA graph scenario)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    max_num_reqs = 32
    num_tokens = 8
    vocab_size = 4096
    dtype = torch.float16

    # Pre-allocate all buffers
    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype)
    original_logits = logits.clone()

    cond_logits_indices = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
    uncond_logits_indices = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
    guidance_scales = torch.ones(max_num_reqs, dtype=torch.float32, device="cuda")

    # Set up 2 actual CFG pairs
    num_cfg_pairs = 2
    cond_logits_indices[0] = 0
    uncond_logits_indices[0] = 1
    guidance_scales[0] = 2.5
    cond_logits_indices[1] = 4
    uncond_logits_indices[1] = 5
    guidance_scales[1] = 1.5

    apply_cfg_logits(
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs,
    )

    # Verify pair 0 (compute in float32, convert to target dtype)
    expected_0 = (
        original_logits[0].float()
        + 2.5 * (original_logits[0].float() - original_logits[1].float())
    ).to(dtype)
    torch.testing.assert_close(logits[0], expected_0, atol=1e-4, rtol=1e-4)

    # Verify pair 1 (compute in float32, convert to target dtype)
    expected_4 = (
        original_logits[4].float()
        + 1.5 * (original_logits[4].float() - original_logits[5].float())
    ).to(dtype)
    torch.testing.assert_close(logits[4], expected_4, atol=1e-4, rtol=1e-4)

    # Other positions unchanged
    torch.testing.assert_close(logits[1], original_logits[1], atol=0, rtol=0)
    torch.testing.assert_close(logits[2], original_logits[2], atol=0, rtol=0)
    torch.testing.assert_close(logits[3], original_logits[3], atol=0, rtol=0)
    torch.testing.assert_close(logits[5], original_logits[5], atol=0, rtol=0)


def test_apply_cfg_logits_large_vocab():
    """Test apply_cfg_logits with large vocabulary size."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_tokens = 4
    vocab_size = 128256  # LLaMA-3 vocab size
    dtype = torch.float16

    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype)
    original_logits = logits.clone()

    cond_logits_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    uncond_logits_indices = torch.tensor([2], dtype=torch.int32, device="cuda")
    guidance_scales = torch.tensor([7.5], dtype=torch.float32, device="cuda")

    apply_cfg_logits(
        logits,
        cond_logits_indices,
        uncond_logits_indices,
        guidance_scales,
        num_cfg_pairs=1,
    )

    # Compute expected in float32, convert to target dtype
    expected = (
        original_logits[0].float()
        + 7.5 * (original_logits[0].float() - original_logits[2].float())
    ).to(dtype)
    torch.testing.assert_close(logits[0], expected, atol=1e-3, rtol=1e-3)


# ============================================================================
# Tests for copy_columns_by_indices kernel
# ============================================================================


@pytest.mark.parametrize("num_rows", [1, 8, 16, 33])
@pytest.mark.parametrize("num_cols", [4, 32, 128])
@pytest.mark.parametrize("dtype", [torch.long, torch.float16, torch.float32])
def test_copy_columns_by_indices_basic(
    num_rows: int, num_cols: int, dtype: torch.dtype
):
    """Test basic copy from source columns to destination columns."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_pairs = 2

    data = torch.arange(num_rows * num_cols, device="cuda", dtype=dtype).reshape(
        num_rows, num_cols
    )
    original = data.clone()

    # Copy column 1 → column 0, column 3 → column 2
    src_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    dst_indices = torch.tensor([0, 2], dtype=torch.int32, device="cuda")

    copy_columns_by_indices(data, src_indices, dst_indices, num_pairs)

    # Verify copied columns
    torch.testing.assert_close(data[:, 0], original[:, 1])
    torch.testing.assert_close(data[:, 2], original[:, 3])

    # Verify untouched columns
    for c in range(num_cols):
        if c not in (0, 2):
            torch.testing.assert_close(data[:, c], original[:, c])


def test_copy_columns_by_indices_no_pairs():
    """Test with zero pairs (should be a no-op)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    data = torch.randn(8, 16, device="cuda")
    original = data.clone()

    # Pre-allocated buffers but zero pairs
    src_indices = torch.zeros(32, dtype=torch.int32, device="cuda")
    dst_indices = torch.zeros(32, dtype=torch.int32, device="cuda")

    copy_columns_by_indices(data, src_indices, dst_indices, num_pairs=0)

    torch.testing.assert_close(data, original, atol=0, rtol=0)


def test_copy_columns_by_indices_preallocated():
    """Test with pre-allocated buffers larger than num_pairs (CUDA graph scenario)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    max_num_reqs = 1024
    num_rows = 8  # num_quantizers
    num_cols = 64  # batch size
    num_pairs = 2

    data = torch.randint(0, 1024, (num_rows, num_cols), device="cuda", dtype=torch.long)
    original = data.clone()

    # Pre-allocated buffers (as in CFGBuffers)
    src_indices = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
    dst_indices = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")

    # Set up 2 actual pairs: cond pos 0→uncond pos 1, cond pos 4→uncond pos 5
    src_indices[0] = 0
    dst_indices[0] = 1
    src_indices[1] = 4
    dst_indices[1] = 5

    copy_columns_by_indices(data, src_indices, dst_indices, num_pairs)

    # Verify copied columns
    torch.testing.assert_close(data[:, 1], original[:, 0])
    torch.testing.assert_close(data[:, 5], original[:, 4])

    # Verify all other columns untouched
    for c in range(num_cols):
        if c not in (1, 5):
            torch.testing.assert_close(data[:, c], original[:, c])


def test_copy_columns_by_indices_single_pair():
    """Test with a single CFG pair."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_rows = 8
    num_cols = 16
    data = torch.randn(num_rows, num_cols, device="cuda")
    original = data.clone()

    src_indices = torch.tensor([3], dtype=torch.int32, device="cuda")
    dst_indices = torch.tensor([7], dtype=torch.int32, device="cuda")

    copy_columns_by_indices(data, src_indices, dst_indices, num_pairs=1)

    torch.testing.assert_close(data[:, 7], original[:, 3])

    for c in range(num_cols):
        if c != 7:
            torch.testing.assert_close(data[:, c], original[:, c])


@pytest.mark.parametrize("num_pairs", [1, 2, 4])
def test_copy_columns_by_indices_cfg_scenario(num_pairs: int):
    """Test in a realistic CFG scenario (code tensor with quantizer codes)."""
    if not current_platform.is_cuda():
        pytest.skip("CFG ops require CUDA")

    num_quantizers = 8
    batch_size = num_pairs * 2 + 4  # some extra non-CFG requests
    max_num_reqs = 1024

    # Simulate code tensor: (num_quantizers, batch_size)
    code = torch.randint(
        0, 1024, (num_quantizers, batch_size), device="cuda", dtype=torch.long
    )
    original = code.clone()

    # Pre-allocated index buffers
    src_indices = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
    dst_indices = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")

    # Set up pairs: cond at even positions, uncond at odd positions
    for i in range(num_pairs):
        src_indices[i] = i * 2      # cond position
        dst_indices[i] = i * 2 + 1  # uncond position

    copy_columns_by_indices(code, src_indices, dst_indices, num_pairs)

    # Verify: uncond positions should now have cond codes
    for i in range(num_pairs):
        cond_pos = i * 2
        uncond_pos = i * 2 + 1
        torch.testing.assert_close(code[:, uncond_pos], original[:, cond_pos])

    # Cond positions and other positions should be unchanged
    for c in range(batch_size):
        if c % 2 == 0 or c >= num_pairs * 2:
            torch.testing.assert_close(code[:, c], original[:, c])
