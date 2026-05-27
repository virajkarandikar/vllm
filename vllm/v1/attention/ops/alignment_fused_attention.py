# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused Alignment Extraction for Batched Decode

Replaces the per-request Python loop in _maybe_emit_alignment with a single
batched GPU operation.  The main Triton attention kernel is left untouched;
instead a lightweight PyTorch function runs immediately afterward and:

  1. Gathers K for each (request, aligned_head) pair via block_table
  2. Computes Q·K·scale  →  softmax  →  argmax/max  (all on GPU)
  3. Writes per-request text_position / max_attn into GPU buffers

A vectorised state machine (`update_alignment_state_gpu`) then decides
whether to suppress or force EOS, and `apply_alignment_to_logits_gpu`
applies the decision to logits without any CPU transfer.

Toggle: CHATTERBOX_FUSED_ALIGNMENT=1  (default 0 = old CPU path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Aligned heads for T3 / Llama (layer_idx, head_idx)
# Must match LLAMA_ALIGNED_HEADS in custom_triton.py / alignment_stream_analyzer.py
# ---------------------------------------------------------------------------
ALIGNED_HEADS: list[tuple[int, int]] = [(12, 15), (13, 11), (9, 2)]
ALIGNED_LAYERS: set[int] = {l for l, _ in ALIGNED_HEADS}
NUM_ALIGNED_HEADS: int = len(ALIGNED_HEADS)

FUSED_ALIGNMENT_ENABLED: bool = (
    os.environ.get("CHATTERBOX_FUSED_ALIGNMENT", "1") == "1"
)

# Tunable alignment parameters (env-configurable, safe defaults)
ALIGNMENT_COMPLETION_OFFSET: int = int(os.environ.get("CHATTERBOX_ALIGNMENT_COMPLETION_OFFSET", "3"))
ALIGNMENT_TAIL_THRESHOLD: float = float(os.environ.get("CHATTERBOX_ALIGNMENT_TAIL_THRESHOLD", "5.0"))
ALIGNMENT_GRACE_FRAMES: int = int(os.environ.get("CHATTERBOX_ALIGNMENT_GRACE_FRAMES", "0"))
ALIGNMENT_REP_THRESHOLD: float = float(os.environ.get("CHATTERBOX_ALIGNMENT_REP_THRESHOLD", "5.0"))


# ───────────────────────────────────────────────────────────────────────────
# GPU-resident per-request state
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class AlignmentState:
    """Fixed-size GPU tensors indexed by *slot* (not request id)."""

    text_position: torch.Tensor       # [max_batch] int32
    complete: torch.Tensor            # [max_batch] bool
    completed_at: torch.Tensor        # [max_batch] int32  (frame index)
    curr_frame: torch.Tensor          # [max_batch] int32
    force_eos: torch.Tensor           # [max_batch] bool
    suppress_eos: torch.Tensor        # [max_batch] bool
    # running sum of max-attn over last-3 text tokens after completion
    tail_attn_sum: torch.Tensor       # [max_batch] float32
    # accumulator for alignment_repetition proxy: sum of max over non-tail region
    rep_attn_sum: torch.Tensor        # [max_batch] float32
    # last 4 generated tokens per request for repetition detection
    recent_tokens: torch.Tensor       # [max_batch, 4] int32

    @classmethod
    def create(cls, max_batch: int, device: torch.device) -> "AlignmentState":
        return cls(
            text_position=torch.zeros(max_batch, dtype=torch.int32, device=device),
            complete=torch.zeros(max_batch, dtype=torch.bool, device=device),
            completed_at=torch.full((max_batch,), -1, dtype=torch.int32, device=device),
            curr_frame=torch.zeros(max_batch, dtype=torch.int32, device=device),
            force_eos=torch.zeros(max_batch, dtype=torch.bool, device=device),
            suppress_eos=torch.ones(max_batch, dtype=torch.bool, device=device),
            tail_attn_sum=torch.zeros(max_batch, dtype=torch.float32, device=device),
            rep_attn_sum=torch.zeros(max_batch, dtype=torch.float32, device=device),
            recent_tokens=torch.full((max_batch, 4), -1, dtype=torch.int32, device=device),
        )

    def reset_slot(self, slot: int) -> None:
        self.text_position[slot] = 0
        self.complete[slot] = False
        self.completed_at[slot] = -1
        self.curr_frame[slot] = 0
        self.force_eos[slot] = False
        self.suppress_eos[slot] = True
        self.tail_attn_sum[slot] = 0.0
        self.rep_attn_sum[slot] = 0.0
        self.recent_tokens[slot].fill_(-1)


# ───────────────────────────────────────────────────────────────────────────
# Batched alignment extraction  (pure PyTorch — no custom Triton kernel needed)
# ───────────────────────────────────────────────────────────────────────────
def _batched_alignment_extract(
    query: torch.Tensor,          # [num_tokens, num_query_heads, head_size]
    key_cache: torch.Tensor,      # [num_blks, blk_size, num_kv_heads, head_size]
    block_table: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,       # [num_seqs]
    cu_seqlens_q: torch.Tensor,   # [num_seqs+1]
    head_indices: list[int],      # query-head indices for aligned heads
    kv_head_indices: list[int],   # kv-head indices for aligned heads
    text_starts: torch.Tensor,    # [num_seqs] int32 — start of text region
    text_ends: torch.Tensor,      # [num_seqs] int32 — end of text region
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (text_positions, max_attns, full_seq_attns) on GPU.

    text_positions : [num_seqs] int32  — argmax over text region (averaged heads)
    max_attns      : [num_seqs] float32 — max attention in text region
    full_seq_attns : [num_seqs, max_seq_len] float32 — full attention for state checks
    """
    num_seqs = seq_lens.numel()
    device = query.device
    num_heads = len(head_indices)
    block_size = key_cache.shape[1]
    head_size = key_cache.shape[3]
    max_seq_len = int(seq_lens.max().item())

    # Pre-allocate outputs
    text_positions = torch.zeros(num_seqs, dtype=torch.int32, device=device)
    max_attns = torch.zeros(num_seqs, dtype=torch.float32, device=device)
    full_seq_attns = torch.zeros(num_seqs, max_seq_len, dtype=torch.float32, device=device)

    for b in range(num_seqs):
        sl = int(seq_lens[b].item())
        if sl == 0:
            continue
        ts = int(text_starts[b].item())
        te = int(text_ends[b].item())
        if te <= ts:
            continue

        num_blocks = (sl + block_size - 1) // block_size
        blk_ids = block_table[b, :num_blocks].to(torch.int64)
        # q token index for this request (decode = last token in packed layout)
        q_idx = int(cu_seqlens_q[b + 1].item()) - 1

        # Accumulate softmax-attention across aligned heads
        avg_attn = torch.zeros(sl, dtype=torch.float32, device=device)
        for hi, kvi in zip(head_indices, kv_head_indices):
            q_vec = query[q_idx, hi]                     # [head_size]
            k_blocks = key_cache[blk_ids, :, kvi, :]     # [nb, bs, hs]
            k_flat = k_blocks.reshape(-1, head_size)[:sl] # [sl, hs]
            scores = (q_vec @ k_flat.T) * scale           # [sl]
            attn = torch.softmax(scores.float(), dim=-1)  # [sl]
            avg_attn += attn

        avg_attn /= num_heads
        full_seq_attns[b, :sl] = avg_attn

        text_attn = avg_attn[ts:te]
        text_positions[b] = int(text_attn.argmax().item())
        max_attns[b] = text_attn.max()

    return text_positions, max_attns, full_seq_attns


# ───────────────────────────────────────────────────────────────────────────
# Vectorised GPU state update  (replaces AlignmentStreamAnalyzer.step)
# ───────────────────────────────────────────────────────────────────────────
def update_alignment_state_gpu(
    state: AlignmentState,
    active_mask: torch.Tensor,       # [max_batch] bool — which slots are live
    text_positions: torch.Tensor,    # [num_active] int32 — argmax results
    max_attns: torch.Tensor,         # [num_active] float32
    full_attns: torch.Tensor,        # [num_active, max_seq] float32
    text_starts: torch.Tensor,       # [num_active] int32
    text_ends: torch.Tensor,         # [num_active] int32
    active_slots: torch.Tensor,      # [num_active] int64 — slot indices
    next_tokens: torch.Tensor | None,  # [num_active] int32 — latest tokens
) -> None:
    """Update state for active requests entirely on GPU."""
    if active_slots.numel() == 0:
        return

    s = active_slots
    n = s.numel()

    # Advance frame counter
    state.curr_frame[s] += 1

    # --- position update with discontinuity guard ---
    old_pos = state.text_position[s]
    delta = text_positions - old_pos
    valid = (delta > -4) & (delta < 7)
    state.text_position[s] = torch.where(valid, text_positions, old_pos)

    # --- completion detection (offset configurable via CHATTERBOX_ALIGNMENT_COMPLETION_OFFSET) ---
    text_lens = text_ends - text_starts
    c_off = ALIGNMENT_COMPLETION_OFFSET
    newly_complete = (state.text_position[s] >= text_lens - c_off) & ~state.complete[s]
    state.complete[s] = state.complete[s] | newly_complete
    state.completed_at[s] = torch.where(
        newly_complete, state.curr_frame[s], state.completed_at[s]
    )

    # --- EOS suppression: suppress until near end of text ---
    state.suppress_eos[s] = (state.text_position[s] < text_lens - c_off) & (text_lens > 5)

    # --- long-tail detection (attention on last c_off text tokens after completion) ---
    for i in range(n):
        slot = int(s[i].item())
        if not state.complete[slot]:
            continue
        te = int(text_ends[i].item())
        ts = int(text_starts[i].item())
        tl_ = te - ts
        if tl_ < c_off:
            continue
        # Sum of attention on last c_off text tokens
        tail_attn = full_attns[i, te - c_off:te].sum()
        state.tail_attn_sum[slot] += tail_attn.item()
        # Sum of max attention on non-tail region (repetition proxy)
        if tl_ > 5:
            non_tail = full_attns[i, ts:te - 5]
            if non_tail.numel() > 0:
                state.rep_attn_sum[slot] += non_tail.max().item()

    # --- token repetition tracking ---
    if next_tokens is not None:
        for i in range(n):
            slot = int(s[i].item())
            tok = int(next_tokens[i].item())
            if tok >= 0:
                state.recent_tokens[slot] = torch.roll(state.recent_tokens[slot], -1)
                state.recent_tokens[slot, -1] = tok

    # --- force EOS decision (vectorised, thresholds configurable) ---
    long_tail = state.complete[s] & (state.tail_attn_sum[s] >= ALIGNMENT_TAIL_THRESHOLD)
    alignment_rep = state.complete[s] & (state.rep_attn_sum[s] > ALIGNMENT_REP_THRESHOLD)
    # 3-of-last-3 same token
    rt = state.recent_tokens[s]  # [n, 4]
    tok_rep = (rt[:, -1] >= 0) & (rt[:, -1] == rt[:, -2]) & (rt[:, -2] == rt[:, -3])

    # Grace period: don't force EOS until N frames after completion
    grace_ok = (state.curr_frame[s] - state.completed_at[s]) >= ALIGNMENT_GRACE_FRAMES
    state.force_eos[s] = grace_ok & (long_tail | alignment_rep | tok_rep)


# ───────────────────────────────────────────────────────────────────────────
# Vectorised logits modification
# ───────────────────────────────────────────────────────────────────────────
def apply_alignment_to_logits_gpu(
    logits: torch.Tensor,            # [batch, vocab]
    state: AlignmentState,
    active_slots: torch.Tensor,      # [batch] int64 — mapping batch pos → slot
    eos_token_id: int,
) -> torch.Tensor:
    """Suppress or force EOS in logits based on GPU alignment state."""
    batch = logits.shape[0]
    if active_slots.numel() < batch:
        return logits

    slots = active_slots[:batch]

    # Suppress EOS for requests not yet near text end
    suppress = state.suppress_eos[slots]  # [batch] bool
    if suppress.any():
        logits[suppress, eos_token_id] = -(2**15)

    # Force EOS for requests with detected issues
    force = state.force_eos[slots]  # [batch] bool
    if force.any():
        forced = torch.full_like(logits[force], -(2**15))
        forced[:, eos_token_id] = 2**15
        logits[force] = forced

    return logits


# ───────────────────────────────────────────────────────────────────────────
# Manager that owns state + orchestrates extraction & update
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class FusedAlignmentManager:
    """Drop-in replacement for per-request AlignmentStreamAnalyzer instances."""

    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    max_batch: int = 64
    state: AlignmentState | None = None

    # Per-slot registration: slot → (text_start, text_end)
    _slot_text_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)
    # req_id → slot mapping (stable across vLLM reordering)
    _req_to_slot: dict[str, int] = field(default_factory=dict)
    _next_slot: int = 0
    _eos_token_id: int = 0
    _num_queries_per_kv: int = 1
    _scale: float = 1.0

    def initialize(
        self,
        eos_token_id: int,
        num_queries_per_kv: int,
        scale: float,
    ) -> None:
        self.state = AlignmentState.create(self.max_batch, self.device)
        self._eos_token_id = eos_token_id
        self._num_queries_per_kv = num_queries_per_kv
        self._scale = scale

    # ── request lifecycle ──────────────────────────────────────────────

    def register_request(self, req_id: str, text_start: int, text_end: int) -> int:
        """Assign a slot and record text region.  Returns slot index."""
        if req_id in self._req_to_slot:
            return self._req_to_slot[req_id]
        slot = self._next_slot
        self._next_slot += 1
        if self._next_slot >= self.max_batch:
            self._next_slot = 0  # wrap (safe: old slots will have been unregistered)
        self._req_to_slot[req_id] = slot
        self._slot_text_ranges[slot] = (text_start, text_end)
        if self.state is not None:
            self.state.reset_slot(slot)
        print(f"[FusedAlignment] REGISTER req={req_id} slot={slot} text=({text_start},{text_end})")
        return slot

    def unregister_request(self, req_id: str) -> None:
        slot = self._req_to_slot.pop(req_id, None)
        if slot is not None:
            self._slot_text_ranges.pop(slot, None)
            if self.state is not None:
                self.state.reset_slot(slot)

    def get_slot(self, req_id: str) -> int | None:
        return self._req_to_slot.get(req_id)

    # ── main entry: extract + update + apply (called once per decode step) ──

    def step(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        req_ids: list[str],
        logits: torch.Tensor,
        next_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """Run extraction + state update + logits modification.

        Called from T3.compute_logits after the main attention pass.
        """
        if self.state is None:
            return logits

        num_seqs = len(req_ids)
        if num_seqs == 0:
            return logits

        # Build active slot list & text ranges for current batch
        active_slots_list: list[int] = []
        ts_list: list[int] = []
        te_list: list[int] = []
        for rid in req_ids:
            slot = self._req_to_slot.get(rid)
            if slot is None or slot not in self._slot_text_ranges:
                active_slots_list.append(-1)
                ts_list.append(0)
                te_list.append(0)
            else:
                active_slots_list.append(slot)
                t = self._slot_text_ranges[slot]
                ts_list.append(t[0])
                te_list.append(t[1])

        active_slots = torch.tensor(active_slots_list, dtype=torch.int64, device=self.device)
        text_starts = torch.tensor(ts_list, dtype=torch.int32, device=self.device)
        text_ends = torch.tensor(te_list, dtype=torch.int32, device=self.device)

        # Which slots are actually registered (not -1)
        valid_mask = active_slots >= 0
        if not valid_mask.any():
            return logits

        # Compute aligned-head indices
        head_indices = [h for _, h in ALIGNED_HEADS]
        kv_head_indices = [h // self._num_queries_per_kv for _, h in ALIGNED_HEADS]

        # Batched extraction — all on GPU, one pass
        text_positions, max_attns, full_attns = _batched_alignment_extract(
            query=query,
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            head_indices=head_indices,
            kv_head_indices=kv_head_indices,
            text_starts=text_starts,
            text_ends=text_ends,
            scale=self._scale,
        )

        # Prepare next_tokens tensor
        nt_tensor = None
        if next_tokens is not None:
            nt_tensor = torch.tensor(next_tokens[:num_seqs], dtype=torch.int32, device=self.device)

        # Filter to valid slots only
        valid_indices = torch.where(valid_mask)[0]
        valid_slots = active_slots[valid_indices]

        # Update state
        update_alignment_state_gpu(
            state=self.state,
            active_mask=valid_mask,
            text_positions=text_positions[valid_indices],
            max_attns=max_attns[valid_indices],
            full_attns=full_attns[valid_indices],
            text_starts=text_starts[valid_indices],
            text_ends=text_ends[valid_indices],
            active_slots=valid_slots,
            next_tokens=nt_tensor[valid_indices] if nt_tensor is not None else None,
        )

        # Apply to logits
        logits = apply_alignment_to_logits_gpu(
            logits=logits,
            state=self.state,
            active_slots=active_slots,
            eos_token_id=self._eos_token_id,
        )

        return logits
