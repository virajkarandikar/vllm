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
from typing import Optional

import torch


@dataclass
class CFGMetadata:
    """
    Pre-allocated metadata for Classifier Free Guidance (CFG).

    This metadata enables:
    1. Zeroing out embeddings for unconditional requests during prefill
    2. Combining cond/uncond logits using the guidance scale formula

    All tensors are pre-allocated to support CUDA graphs. The `num_cfg_pairs`
    field indicates how many entries in the tensors are valid.
    """

    # Number of valid CFG pairs in this batch (rest of tensors may be padding)
    num_cfg_pairs: int

    # Pre-allocated tensor of shape (max_num_reqs,) containing the batch
    # indices of conditional requests for each CFG pair
    # Only first `num_cfg_pairs` entries are valid
    cond_req_indices: torch.Tensor

    # Pre-allocated tensor of shape (max_num_reqs,) containing the batch
    # indices of unconditional requests for each CFG pair
    # Only first `num_cfg_pairs` entries are valid
    uncond_req_indices: torch.Tensor

    # Pre-allocated tensor of shape (max_num_reqs,) containing guidance
    # scale for each CFG pair
    # Only first `num_cfg_pairs` entries are valid
    guidance_scales: torch.Tensor

    # Pre-allocated tensor of shape (max_num_tokens,) - boolean mask
    # True for tokens belonging to unconditional requests
    # Used to zero embeddings during prefill
    uncond_token_mask: torch.Tensor

    # Number of valid tokens in uncond_token_mask
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
        self.cond_req_indices = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.uncond_req_indices = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.guidance_scales = torch.ones(
            max_num_reqs, dtype=torch.float32, device=device
        )
        self.uncond_token_mask = torch.zeros(
            max_num_tokens, dtype=torch.bool, device=device
        )

        # CPU tensors for building metadata (pinned for fast H2D transfer)
        self.cond_req_indices_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.uncond_req_indices_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.guidance_scales_cpu = torch.ones(
            max_num_reqs, dtype=torch.float32, pin_memory=pin_memory
        )
        self.uncond_token_mask_cpu = torch.zeros(
            max_num_tokens, dtype=torch.bool, pin_memory=pin_memory
        )

        # Current valid count
        self.num_cfg_pairs = 0
        self.num_tokens = 0

    def reset(self) -> None:
        """Reset the buffers for a new batch."""
        self.num_cfg_pairs = 0
        self.num_tokens = 0
        # Zero out the mask (important for correctness)
        if self.num_tokens > 0:
            self.uncond_token_mask_cpu[: self.num_tokens].zero_()

    def add_cfg_pair(
        self,
        cond_req_idx: int,
        uncond_req_idx: int,
        guidance_scale: float,
    ) -> None:
        """Add a CFG pair to the buffers."""
        idx = self.num_cfg_pairs
        self.cond_req_indices_cpu[idx] = cond_req_idx
        self.uncond_req_indices_cpu[idx] = uncond_req_idx
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
            self.cond_req_indices[:n].copy_(
                self.cond_req_indices_cpu[:n], non_blocking=True
            )
            self.uncond_req_indices[:n].copy_(
                self.uncond_req_indices_cpu[:n], non_blocking=True
            )
            self.guidance_scales[:n].copy_(
                self.guidance_scales_cpu[:n], non_blocking=True
            )
        if self.num_tokens > 0:
            self.uncond_token_mask[: self.num_tokens].copy_(
                self.uncond_token_mask_cpu[: self.num_tokens], non_blocking=True
            )

    def get_metadata(self) -> Optional[CFGMetadata]:
        """Get CFGMetadata from the current buffer state.

        Returns None if there are no CFG pairs.
        """
        if self.num_cfg_pairs == 0:
            return None

        return CFGMetadata(
            num_cfg_pairs=self.num_cfg_pairs,
            cond_req_indices=self.cond_req_indices,
            uncond_req_indices=self.uncond_req_indices,
            guidance_scales=self.guidance_scales,
            uncond_token_mask=self.uncond_token_mask,
            num_tokens=self.num_tokens,
        )
