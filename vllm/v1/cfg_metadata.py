# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CFG (Classifier Free Guidance) metadata for vLLM v1.

This module provides metadata structures for implementing CFG during inference.
CFG requires running both conditional and unconditional forward passes and
combining their logits using:
    logits_out = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

The metadata is designed for CUDA graph compatibility with pre-allocated tensors.
"""

from dataclasses import dataclass

import torch


@dataclass
class CFGMetadata:
    """
    Pre-allocated metadata for Classifier Free Guidance (CFG).

    This metadata enables:
    1. Zeroing out embeddings for unconditional requests during prefill
       (uses uncond_token_mask)
    2. Combining cond/uncond logits using the guidance scale formula
       (uses cond_logits_indices, uncond_logits_indices, guidance_scales)

    All tensors are pre-allocated to support CUDA graphs. The `num_cfg_pairs`
    field indicates how many entries in the tensors are valid.

    For prefill: logits indices point to the last token of each request's
    prefill chunk (careful with chunked prefill - only the final chunk
    produces samplable logits).

    For decode: each request has one token, and its index in the packed
    tensor is what goes in the metadata.
    """

    # Pre-allocated tensor of shape (max_num_reqs,) containing guidance
    # scale for each CFG pair
    # Only first `num_cfg_pairs` entries are valid
    guidance_scales: torch.Tensor

    # Pre-allocated tensor of shape (max_num_tokens,) - boolean mask
    # True for tokens belonging to unconditional requests
    # Used to zero embeddings during prefill
    uncond_token_mask: torch.Tensor

    # ---- Token-level indices for CFG logits application ----
    # These are token positions in the packed logits tensor (seq_len, vocab_size)
    # for the sampling position of each CFG pair.
    # Used by apply_cfg_logits kernel.

    # Pre-allocated tensor of shape (max_num_reqs,) containing the token
    # indices (in packed logits) for conditional requests' sampling positions
    # Only first `num_cfg_pairs` entries are valid
    cond_logits_indices: torch.Tensor

    # Pre-allocated tensor of shape (max_num_reqs,) containing the token
    # indices (in packed logits) for unconditional requests' sampling positions
    # Only first `num_cfg_pairs` entries are valid
    uncond_logits_indices: torch.Tensor

    # Number of valid CFG pairs in this batch (rest of tensors may be padding)
    num_cfg_pairs: int = 0

    # Number of valid tokens in uncond_token_mask. Used to zero embeddings during prefill.
    num_tokens: int = 0


class CFGBuffers:
    """
    Pre-allocated buffers for CFG metadata, compatible with CUDA graphs.

    These buffers are allocated once during model runner initialization
    and reused each step. The actual valid data size is tracked separately.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
        pin_memory: bool = True,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        # GPU tensors (pre-allocated for CUDA graphs)
        self.guidance_scales = torch.ones(
            max_num_reqs, dtype=torch.float32, device=device
        )
        self.uncond_token_mask = torch.zeros(
            max_num_tokens, dtype=torch.bool, device=device
        )

        # Token-level indices for CFG logits application
        # These store the token positions in packed logits for sampling
        self.cond_logits_indices = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.uncond_logits_indices = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # CPU tensors for building metadata (pinned for fast H2D transfer)
        self.guidance_scales_cpu = torch.ones(
            max_num_reqs, dtype=torch.float32, pin_memory=pin_memory
        )
        self.uncond_token_mask_cpu = torch.zeros(
            max_num_tokens, dtype=torch.bool, pin_memory=pin_memory
        )
        self.cond_logits_indices_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.uncond_logits_indices_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )

        # Current valid count
        self.num_cfg_pairs = 0
        self.num_tokens = 0

    def reset(self) -> None:
        """Reset the buffers for a new batch."""
        self.num_cfg_pairs = 0
        # Zero out the mask (important for correctness)
        if self.num_tokens > 0:
            self.uncond_token_mask_cpu[: self.num_tokens].zero_()
        self.num_tokens = 0

    def add_cfg_pair(
        self,
        cond_logits_idx: int,
        uncond_logits_idx: int,
        guidance_scale: float,
    ) -> None:
        """Add a CFG pair to the buffers.

        Args:
            cond_logits_idx: Token index (in packed logits) for conditional
                            request's sampling position. For prefill, this is
                            the last token of the prefill chunk. For decode,
                            this is the single token's position in the packed
                            tensor.
            uncond_logits_idx: Token index (in packed logits) for unconditional
                              request's sampling position.
            guidance_scale: Guidance scale for this CFG pair.
        """
        idx = self.num_cfg_pairs
        self.cond_logits_indices_cpu[idx] = cond_logits_idx
        self.uncond_logits_indices_cpu[idx] = uncond_logits_idx
        self.guidance_scales_cpu[idx] = guidance_scale
        self.num_cfg_pairs += 1

    def set_uncond_token_range(self, start: int, end: int) -> None:
        """Mark a range of tokens as belonging to unconditional requests."""
        self.uncond_token_mask_cpu[start:end] = True
        self.num_tokens = max(self.num_tokens, end)

    def sync_to_gpu(self) -> None:
        """Copy CPU buffers to GPU (async, non-blocking)."""
        if self.num_cfg_pairs > 0:
            n = self.num_cfg_pairs
            self.guidance_scales[:n].copy_(
                self.guidance_scales_cpu[:n], non_blocking=True
            )
            self.cond_logits_indices[:n].copy_(
                self.cond_logits_indices_cpu[:n], non_blocking=True
            )
            self.uncond_logits_indices[:n].copy_(
                self.uncond_logits_indices_cpu[:n], non_blocking=True
            )
        if self.num_tokens > 0:
            self.uncond_token_mask[: self.num_tokens].copy_(
                self.uncond_token_mask_cpu[: self.num_tokens], non_blocking=True
            )

    def get_metadata(self) -> CFGMetadata:
        """Get CFGMetadata from the current buffer state.

        Always returns valid metadata. If there are no CFG pairs,
        num_cfg_pairs will be 0 and kernels will early-exit.
        This is required for CUDA graph compatibility.
        """
        return CFGMetadata(
            guidance_scales=self.guidance_scales,
            uncond_token_mask=self.uncond_token_mask,
            cond_logits_indices=self.cond_logits_indices,
            uncond_logits_indices=self.uncond_logits_indices,
            num_cfg_pairs=self.num_cfg_pairs,
            num_tokens=self.num_tokens,
        )
