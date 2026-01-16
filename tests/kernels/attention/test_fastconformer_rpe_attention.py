# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FastConformer Relative Position Encoding (RPE) attention kernel.

This kernel implements multi-head attention with relative positional encoding
as described in Transformer-XL (https://arxiv.org/abs/1901.02860).
"""

import math

import pytest
import torch
import torch.nn as nn

from vllm.v1.attention.backends.fastconformer_rpe_attention import (
    FastConformerRPEMetadata,
    _launch_fc_fused_cache_triton,
)


class RelPositionMultiHeadAttentionReference(nn.Module):
    """Reference implementation of RelPositionMultiHeadAttention from NeMo.

    This is the PyTorch reference implementation for testing the Triton kernel.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head: Number of attention heads.
        n_feat: Size of features (d_model).
        window: Attention window size (W).
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        window: int,
    ):
        super().__init__()
        self.h = n_head
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.window = window

        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        # Learnable biases for matrix c and matrix d
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding shift.

        Args:
            x: Tensor of shape (batch, nheads, time, 2*time-1).

        Returns:
            Shifted tensor of shape (batch, nheads, time, 2*time-1).
        """
        b, h, qlen, pos_len = x.size()
        # Add a column of zeros on the left side
        x = torch.nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        # Drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def _compute_rel_pos_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute relative positional embeddings for the sequence.

        The kernel uses delta = query_pos - key_pos (positive means key is in past).
        We create embeddings for deltas from -(W-1) to (W-1), but flip the sign
        to match the kernel's convention after rel_shift.

        Args:
            device: Device to place tensors on.
            dtype: Data type for tensors.
            seq_len: Sequence length.

        Returns:
            Positional embeddings of shape (1, 2*seq_len-1, d_model).
        """
        D = self.h * self.d_k
        W = seq_len

        # Create relative position indices from -(W-1) to (W-1)
        # After rel_shift, position (i, j) will use index (W-1) + (j-i)
        # = (W-1) - (i-j) = (W-1) - delta_kernel
        # So we flip the order to match the kernel's convention
        deltas = torch.arange(W - 1, -(W), -1, device=device)[:, None].to(torch.float32)

        # Sinusoidal positional encoding
        div = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / D)
        )
        sin = torch.sin(deltas * div)
        cos = torch.cos(deltas * div)

        rel = torch.zeros((2 * W - 1, D), device=device, dtype=torch.float32)
        rel[:, 0::2] = sin
        rel[:, 1::2] = cos

        return rel.unsqueeze(0).to(dtype)  # [1, 2*W-1, D]

    def forward(
        self,
        query: torch.Tensor,  # [B, T, D]
        key: torch.Tensor,    # [B, T, D]
        value: torch.Tensor,  # [B, T, D]
        mask: torch.Tensor | None = None,  # [B, T, T] bool mask (True = masked)
    ) -> torch.Tensor:
        """Compute attention with relative positional encoding.

        Args:
            query: Query tensor of shape (batch, time, d_model).
            key: Key tensor of shape (batch, time, d_model).
            value: Value tensor of shape (batch, time, d_model).
            mask: Optional attention mask (True values are masked).

        Returns:
            Output tensor of shape (batch, time, d_model).
        """
        B, T, D = query.shape

        # Reshape to [B, T, H, Dh] then transpose to [B, H, T, Dh]
        q = query.view(B, T, self.h, self.d_k).transpose(1, 2)
        k = key.view(B, T, self.h, self.d_k).transpose(1, 2)
        v = value.view(B, T, self.h, self.d_k).transpose(1, 2)

        # Compute positional embeddings
        pos_emb = self._compute_rel_pos_emb(query.device, query.dtype, T)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # [1, H, 2T-1, Dh]

        # Add biases to query (reshape for broadcasting: [1, H, 1, Dh])
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)  # [B, H, T, Dh]
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)  # [B, H, T, Dh]

        # Matrix AC: content-based attention
        # (B, H, T, Dh) x (B, H, Dh, T) -> (B, H, T, T)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # Matrix BD: position-based attention
        # (B, H, T, Dh) x (1, H, Dh, 2T-1) -> (B, H, T, 2T-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, :T]  # Trim to [B, H, T, T]

        # Combined attention scores
        scores = (matrix_ac + matrix_bd) / self.s_d_k  # [B, H, T, T]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, T, T]
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax and weighted sum
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, H, T, Dh]

        # Reshape back to [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out


def generate_random_qkv(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors.

    Args:
        seq_len: Sequence length.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Device to place tensors on.
        dtype: Data type.
        seed: Random seed.

    Returns:
        Tuple of (query, key, value) tensors, each of shape (1, seq_len, d_model).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    d_model = n_heads * head_dim

    q = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, generator=generator)
    k = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, generator=generator)
    v = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, generator=generator)

    return q, k, v


def generate_random_params(
    n_heads: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, nn.Linear]:
    """Generate random attention parameters.

    Args:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Device to place tensors on.
        dtype: Data type.
        seed: Random seed.

    Returns:
        Tuple of (pos_bias_u, pos_bias_v, linear_pos).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    d_model = n_heads * head_dim

    pos_bias_u = torch.randn(
        n_heads, head_dim, device=device, dtype=dtype, generator=generator
    )
    pos_bias_v = torch.randn(
        n_heads, head_dim, device=device, dtype=dtype, generator=generator
    )

    linear_pos = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
    nn.init.normal_(linear_pos.weight, mean=0, std=0.02, generator=generator)

    return pos_bias_u, pos_bias_v, linear_pos


def create_causal_sliding_window_mask(
    seq_len: int,
    window: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a causal sliding window attention mask.

    Args:
        seq_len: Sequence length.
        window: Window size (positions to look back).
        device: Device to place tensor on.

    Returns:
        Boolean mask of shape (1, seq_len, seq_len) where True = masked.
    """
    # Create position indices
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)

    # Causal mask: can only attend to positions <= current position
    causal_mask = cols > rows

    # Sliding window mask: can only attend to positions within window
    # For position i, can attend to positions max(0, i-window) to i
    window_mask = (rows - cols) > window

    # Combined mask: True means masked (cannot attend)
    mask = causal_mask | window_mask

    return mask.unsqueeze(0)  # [1, T, T]


def simulate_kv_cache(
    key: torch.Tensor,  # [B, T, D]
    value: torch.Tensor,  # [B, T, D]
    block_size: int,
    n_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate KV cache storage for testing.

    Args:
        key: Key tensor of shape (B, T, D).
        value: Value tensor of shape (B, T, D).
        block_size: Block size for KV cache.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Tuple of (key_cache, value_cache, block_table, slot_mapping).
    """
    B, T, D = key.shape
    device = key.device
    dtype = key.dtype

    # Calculate number of blocks needed
    num_blocks = (T + block_size - 1) // block_size

    # Create KV cache tensors [num_blocks, block_size, n_heads, head_dim]
    key_cache = torch.zeros(
        num_blocks * B, block_size, n_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.zeros(
        num_blocks * B, block_size, n_heads, head_dim, device=device, dtype=dtype
    )

    # Reshape key/value to [B, T, H, Dh]
    key_reshaped = key.view(B, T, n_heads, head_dim)
    value_reshaped = value.view(B, T, n_heads, head_dim)

    # Create slot mapping and block table
    slot_mapping = torch.arange(T, device=device, dtype=torch.long)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)

    # Fill KV cache
    for t in range(T):
        block_idx = t // block_size
        offset = t % block_size
        key_cache[block_idx, offset] = key_reshaped[0, t]
        value_cache[block_idx, offset] = value_reshaped[0, t]

    return key_cache, value_cache, block_table, slot_mapping


def compute_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pos_bias_u: torch.Tensor,
    pos_bias_v: torch.Tensor,
    linear_pos: nn.Linear,
    window: int,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute reference attention output using PyTorch implementation.

    Args:
        query: Query tensor of shape (1, T, D).
        key: Key tensor of shape (1, T, D).
        value: Value tensor of shape (1, T, D).
        pos_bias_u: Position bias u of shape (H, Dh).
        pos_bias_v: Position bias v of shape (H, Dh).
        linear_pos: Linear layer for positional encoding.
        window: Attention window size.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Output tensor of shape (1, T, D).
    """
    d_model = n_heads * head_dim

    # Create reference model
    ref_model = RelPositionMultiHeadAttentionReference(
        n_head=n_heads,
        n_feat=d_model,
        window=window,
    ).to(query.device).to(query.dtype)

    # Copy parameters
    ref_model.pos_bias_u.data.copy_(pos_bias_u)
    ref_model.pos_bias_v.data.copy_(pos_bias_v)
    ref_model.linear_pos.weight.data.copy_(linear_pos.weight.data)

    # Create causal sliding window mask
    T = query.shape[1]
    mask = create_causal_sliding_window_mask(T, window, device=query.device)

    # Compute reference output
    with torch.no_grad():
        out_ref = ref_model(query, key, value, mask)

    return out_ref


def compute_kernel_attention(
    query: torch.Tensor,  # [1, T, D]
    key: torch.Tensor,    # [1, T, D]
    value: torch.Tensor,  # [1, T, D]
    pos_bias_u: torch.Tensor,
    pos_bias_v: torch.Tensor,
    linear_pos: nn.Linear,
    window: int,
    n_heads: int,
    head_dim: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Compute attention using the Triton kernel.

    This simulates streaming inference by processing tokens one at a time
    after an initial prompt.

    Args:
        query: Query tensor of shape (1, T, D).
        key: Key tensor of shape (1, T, D).
        value: Value tensor of shape (1, T, D).
        pos_bias_u: Position bias u of shape (H, Dh).
        pos_bias_v: Position bias v of shape (H, Dh).
        linear_pos: Linear layer for positional encoding.
        window: Attention window size.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Block size for KV cache.

    Returns:
        Output tensor of shape (1, T, D).
    """
    B, T, D = query.shape
    device = query.device
    dtype = query.dtype

    # Simulate KV cache
    key_cache, value_cache, block_table, slot_mapping = simulate_kv_cache(
        key, value, block_size, n_heads, head_dim
    )

    # Reshape query to [T, H, Dh]
    query_flat = query.view(T, n_heads, head_dim).contiguous()

    # Compute relative position projection (matching kernel's _get_rel_proj)
    W = window
    deltas = torch.arange(-W, W + 1, device=device)[:, None].to(torch.float32)
    div = torch.exp(
        torch.arange(0, D, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / D)
    )
    sin = torch.sin(deltas * div)
    cos = torch.cos(deltas * div)
    rel = torch.zeros((2 * W + 1, D), device=device, dtype=torch.float32)
    rel[:, 0::2] = sin
    rel[:, 1::2] = cos
    rel = linear_pos(rel.to(dtype))  # [2W+1, D]
    rel = rel.view(2 * W + 1, n_heads, head_dim).permute(1, 2, 0).contiguous()  # [H, Dh, 2W+1]

    # Create metadata
    # query_start_loc: cumulative token counts [0, T] for 1 request
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
    # decode_offset: how many tokens have been processed before (0 for prefill)
    decode_offset = torch.tensor([0], dtype=torch.int32, device=device)
    # doc_ids: maps each token to its request index (all 0 for single request)
    doc_ids = torch.zeros(T, dtype=torch.int32, device=device)

    metadata = FastConformerRPEMetadata(
        num_actual_tokens=T,
        num_reqs=1,
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=block_size,
        doc_ids=doc_ids,
        decode_offset=decode_offset,
        causal=True,
    )

    # Allocate output
    output = torch.empty(T, n_heads, head_dim, device=device, dtype=torch.float32)

    # Launch kernel
    _launch_fc_fused_cache_triton(
        query_flat,
        key_cache,
        value_cache,
        metadata,
        pos_bias_u.contiguous(),
        pos_bias_v.contiguous(),
        rel,
        output,
    )

    # Reshape output to [1, T, D]
    return output.view(1, T, D).to(dtype)


def compute_kernel_attention_streaming(
    query: torch.Tensor,  # [1, T, D]
    key: torch.Tensor,    # [1, T, D]
    value: torch.Tensor,  # [1, T, D]
    pos_bias_u: torch.Tensor,
    pos_bias_v: torch.Tensor,
    linear_pos: nn.Linear,
    window: int,
    n_heads: int,
    head_dim: int,
    prompt_len: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Compute attention using the Triton kernel with streaming simulation.

    Processes prompt_len tokens first, then one token at a time.

    Args:
        query: Query tensor of shape (1, T, D).
        key: Key tensor of shape (1, T, D).
        value: Value tensor of shape (1, T, D).
        pos_bias_u: Position bias u of shape (H, Dh).
        pos_bias_v: Position bias v of shape (H, Dh).
        linear_pos: Linear layer for positional encoding.
        window: Attention window size.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        prompt_len: Number of tokens to process in the first chunk.
        block_size: Block size for KV cache.

    Returns:
        Output tensor of shape (1, T, D).
    """
    B, T, D = query.shape
    device = query.device
    dtype = query.dtype

    # Simulate KV cache (pre-filled with all K/V for simplicity)
    key_cache, value_cache, block_table, _ = simulate_kv_cache(
        key, value, block_size, n_heads, head_dim
    )

    # Compute relative position projection
    W = window
    deltas = torch.arange(-W, W + 1, device=device)[:, None].to(torch.float32)
    div = torch.exp(
        torch.arange(0, D, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / D)
    )
    sin = torch.sin(deltas * div)
    cos = torch.cos(deltas * div)
    rel = torch.zeros((2 * W + 1, D), device=device, dtype=torch.float32)
    rel[:, 0::2] = sin
    rel[:, 1::2] = cos
    rel = linear_pos(rel.to(dtype))
    rel = rel.view(2 * W + 1, n_heads, head_dim).permute(1, 2, 0).contiguous()

    output_list = []
    i = 0

    while i < T:
        chunk_size = prompt_len if i == 0 else 1
        chunk_size = min(chunk_size, T - i)

        # Extract query chunk
        query_chunk = query[:, i : i + chunk_size, :].view(
            chunk_size, n_heads, head_dim
        ).contiguous()

        # Create metadata for this chunk
        query_start_loc = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)
        decode_offset = torch.tensor([i], dtype=torch.int32, device=device)
        doc_ids = torch.zeros(chunk_size, dtype=torch.int32, device=device)

        # Slot mapping for this chunk
        slot_mapping = torch.arange(i, i + chunk_size, device=device, dtype=torch.long)

        metadata = FastConformerRPEMetadata(
            num_actual_tokens=chunk_size,
            num_reqs=1,
            query_start_loc=query_start_loc,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=block_size,
            doc_ids=doc_ids,
            decode_offset=decode_offset,
            causal=True,
        )

        # Allocate output for this chunk
        output_chunk = torch.empty(
            chunk_size, n_heads, head_dim, device=device, dtype=torch.float32
        )

        # Launch kernel
        _launch_fc_fused_cache_triton(
            query_chunk,
            key_cache,
            value_cache,
            metadata,
            pos_bias_u.contiguous(),
            pos_bias_v.contiguous(),
            rel,
            output_chunk,
        )

        output_list.append(output_chunk.clone())
        i += chunk_size

    # Concatenate outputs
    output = torch.cat(output_list, dim=0)
    return output.view(1, T, D).to(dtype)


@pytest.mark.parametrize("seq_len", [8, 32, 64])
@pytest.mark.parametrize("n_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rpe_attention_vs_reference(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test that the Triton kernel matches the reference PyTorch implementation.

    Args:
        seq_len: Sequence length.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype: Data type for computation.
    """
    device = "cuda"
    window = 71  # Matching kernel's W value
    d_model = n_heads * head_dim

    # Skip if sequence is longer than window (kernel assumes seq_len <= W+1)
    if seq_len > window + 1:
        pytest.skip(f"seq_len {seq_len} > window+1 {window + 1}")

    # Generate random inputs
    query, key, value = generate_random_qkv(
        seq_len=seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Generate random parameters
    pos_bias_u, pos_bias_v, linear_pos = generate_random_params(
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Compute reference output
    out_ref = compute_reference_attention(
        query, key, value,
        pos_bias_u, pos_bias_v, linear_pos,
        window, n_heads, head_dim,
    )

    # Compute kernel output
    out_kernel = compute_kernel_attention(
        query, key, value,
        pos_bias_u, pos_bias_v, linear_pos,
        window, n_heads, head_dim,
    )

    # Compare results
    rtol = 1e-2
    atol = 1e-2

    assert out_kernel.shape == out_ref.shape, (
        f"Shape mismatch: kernel {out_kernel.shape} vs ref {out_ref.shape}"
    )

    max_diff = (out_kernel - out_ref).abs().max().item()
    mean_diff = (out_kernel - out_ref).abs().mean().item()

    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"RPE attention kernel output doesn't match reference. "
        f"seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}, "
        f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
    )


@pytest.mark.parametrize("prompt_len", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [8, 32, 48])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rpe_attention_streaming(
    prompt_len: int,
    seq_len: int,
    dtype: torch.dtype,
):
    """Test streaming (chunked) processing matches full sequence processing.

    Args:
        prompt_len: Number of tokens in the initial prompt chunk.
        seq_len: Total sequence length.
        dtype: Data type for computation.
    """
    device = "cuda"
    n_heads = 8
    head_dim = 64
    window = 71

    if seq_len > window + 1:
        pytest.skip(f"seq_len {seq_len} > window+1 {window + 1}")

    # Generate random inputs
    query, key, value = generate_random_qkv(
        seq_len=seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Generate random parameters
    pos_bias_u, pos_bias_v, linear_pos = generate_random_params(
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Compute full sequence output
    out_full = compute_kernel_attention(
        query, key, value,
        pos_bias_u, pos_bias_v, linear_pos,
        window, n_heads, head_dim,
    )

    # Compute streaming output
    out_streaming = compute_kernel_attention_streaming(
        query, key, value,
        pos_bias_u, pos_bias_v, linear_pos,
        window, n_heads, head_dim,
        prompt_len=prompt_len,
    )

    # Compare results
    rtol = 1e-4
    atol = 1e-4

    max_diff = (out_full - out_streaming).abs().max().item()
    mean_diff = (out_full - out_streaming).abs().mean().item()

    assert torch.allclose(out_full, out_streaming, rtol=rtol, atol=atol), (
        f"Streaming output doesn't match full sequence output. "
        f"prompt_len={prompt_len}, seq_len={seq_len}, "
        f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
    )


@pytest.mark.parametrize("n_seqs", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rpe_attention_batched(
    n_seqs: int,
    dtype: torch.dtype,
):
    """Test batched processing with multiple sequences.

    Args:
        n_seqs: Number of sequences to process together.
        dtype: Data type for computation.
    """
    device = "cuda"
    n_heads = 8
    head_dim = 64
    window = 71
    d_model = n_heads * head_dim
    block_size = 128

    # Generate sequences with varying lengths
    seq_lens = [8 + i * 4 for i in range(n_seqs)]

    # Generate random parameters (shared across sequences)
    pos_bias_u, pos_bias_v, linear_pos = generate_random_params(
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Compute relative position projection (shared)
    W = window
    deltas = torch.arange(-W, W + 1, device=device)[:, None].to(torch.float32)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    sin = torch.sin(deltas * div)
    cos = torch.cos(deltas * div)
    rel = torch.zeros((2 * W + 1, d_model), device=device, dtype=torch.float32)
    rel[:, 0::2] = sin
    rel[:, 1::2] = cos
    rel = linear_pos(rel.to(dtype))
    rel = rel.view(2 * W + 1, n_heads, head_dim).permute(1, 2, 0).contiguous()

    # Generate inputs and compute reference for each sequence
    queries = []
    keys = []
    values = []
    ref_outputs = []

    for i, seq_len in enumerate(seq_lens):
        q, k, v = generate_random_qkv(
            seq_len=seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            seed=42 + i,
        )
        queries.append(q)
        keys.append(k)
        values.append(v)

        # Compute reference output
        ref_out = compute_reference_attention(
            q, k, v,
            pos_bias_u, pos_bias_v, linear_pos,
            window, n_heads, head_dim,
        )
        ref_outputs.append(ref_out)

    # Now compute batched kernel output
    total_tokens = sum(seq_lens)

    # Pack all queries, keys, values
    query_packed = torch.cat([q.view(-1, d_model) for q in queries], dim=0)
    key_packed = torch.cat([k.view(-1, d_model) for k in keys], dim=0)
    value_packed = torch.cat([v.view(-1, d_model) for v in values], dim=0)

    # Create KV cache for all sequences
    max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens)
    total_blocks = n_seqs * max_blocks_per_seq

    key_cache = torch.zeros(
        total_blocks, block_size, n_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.zeros(
        total_blocks, block_size, n_heads, head_dim, device=device, dtype=dtype
    )

    # Fill KV cache and build block table
    block_table = torch.zeros(n_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    slot_mapping_list = []
    block_idx = 0

    for seq_idx, seq_len in enumerate(seq_lens):
        k = keys[seq_idx].view(seq_len, n_heads, head_dim)
        v = values[seq_idx].view(seq_len, n_heads, head_dim)
        num_blocks = (seq_len + block_size - 1) // block_size

        for b in range(num_blocks):
            block_table[seq_idx, b] = block_idx
            start_t = b * block_size
            end_t = min(start_t + block_size, seq_len)
            key_cache[block_idx, : end_t - start_t] = k[start_t:end_t]
            value_cache[block_idx, : end_t - start_t] = v[start_t:end_t]
            block_idx += 1

        # Slot mapping for this sequence
        for t in range(seq_len):
            slot_mapping_list.append(t)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)

    # Build query_start_loc and decode_offset
    query_start_loc_list = [0]
    cumsum = 0
    for seq_len in seq_lens:
        cumsum += seq_len
        query_start_loc_list.append(cumsum)
    query_start_loc = torch.tensor(query_start_loc_list, dtype=torch.int32, device=device)
    decode_offset = torch.zeros(n_seqs, dtype=torch.int32, device=device)

    # Build doc_ids: maps each token to its sequence index
    doc_ids = torch.cat([
        torch.full((seq_len,), seq_idx, dtype=torch.int32, device=device)
        for seq_idx, seq_len in enumerate(seq_lens)
    ])

    # Reshape packed query
    query_flat = query_packed.view(total_tokens, n_heads, head_dim).contiguous()

    # Create metadata
    metadata = FastConformerRPEMetadata(
        num_actual_tokens=total_tokens,
        num_reqs=n_seqs,
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=block_size,
        doc_ids=doc_ids,
        decode_offset=decode_offset,
        causal=True,
    )

    # Allocate output
    output = torch.empty(total_tokens, n_heads, head_dim, device=device, dtype=torch.float32)

    # Launch kernel
    _launch_fc_fused_cache_triton(
        query_flat,
        key_cache,
        value_cache,
        metadata,
        pos_bias_u.contiguous(),
        pos_bias_v.contiguous(),
        rel,
        output,
    )

    # Split output and compare with references
    output = output.to(dtype)
    offset = 0
    rtol = 1e-2
    atol = 1e-2

    for seq_idx, seq_len in enumerate(seq_lens):
        out_seq = output[offset : offset + seq_len].view(1, seq_len, d_model)
        ref_seq = ref_outputs[seq_idx]

        max_diff = (out_seq - ref_seq).abs().max().item()
        mean_diff = (out_seq - ref_seq).abs().mean().item()

        assert torch.allclose(out_seq, ref_seq, rtol=rtol, atol=atol), (
            f"Seq {seq_idx}: Batched output doesn't match reference. "
            f"n_seqs={n_seqs}, seq_len={seq_len}, "
            f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
        )

        offset += seq_len


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rpe_attention_dtypes(dtype: torch.dtype):
    """Test that the kernel works with different data types.

    Args:
        dtype: Data type for computation.
    """
    device = "cuda"
    seq_len = 16
    n_heads = 8
    head_dim = 64
    window = 71

    # Generate random inputs
    query, key, value = generate_random_qkv(
        seq_len=seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Generate random parameters
    pos_bias_u, pos_bias_v, linear_pos = generate_random_params(
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Compute kernel output (should not raise)
    out_kernel = compute_kernel_attention(
        query, key, value,
        pos_bias_u, pos_bias_v, linear_pos,
        window, n_heads, head_dim,
    )

    # Basic sanity checks
    assert out_kernel.shape == query.shape
    assert not torch.isnan(out_kernel).any(), "Output contains NaN values"
    assert not torch.isinf(out_kernel).any(), "Output contains Inf values"

