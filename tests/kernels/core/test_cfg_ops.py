# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CFG (Classifier Free Guidance) operations.

Run `pytest tests/kernels/core/test_cfg_ops.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.cfg_ops import set_uncond_embeddings
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
