"""Eviction policies for a fixed-capacity KV cache.

Every policy sees the same signal: for each generation step (or prefill chunk),
the attention received by each cached token from the most recent queries
(summed over the chunk's query rows, averaged over heads). Policies differ only
in how they turn that signal into an eviction decision. This mirrors the setup
in the paper's Table 4: trace = cached token's K/V vectors, fast store = GPU
memory, value = running attention received.

Interface:
    register(positions, step, attn_received)
        Called once per forward pass with the newly added tokens' absolute
        positions, the current step index, and the attention those new tokens
        received (summed over the chunk's query rows).
    update_old(attn_old, step)
        Called with attention received by the *already cached* tokens from the
        current chunk's queries (length == number of cached tokens).
    choose_evict(n_evict) -> list[int]
        Local indices (0..len-1 over the current cache) to evict. Must never
        include indices of tokens added in the current step (protected), and
        for sink-style policies must respect the sink set.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np


POLICIES = ["full", "random", "window", "streaming", "h2o", "pes"]


class EvictionPolicy:
    name = "base"

    def __init__(self, capacity: Optional[int] = None, seed: int = 0):
        self.capacity = capacity
        self.rng = random.Random(seed)
        # per-token bookkeeping, aligned with the cache's token order
        self.positions: List[int] = []   # absolute original token positions
        self.entry_step: List[int] = []  # step index at which token entered cache
        self.new_start: int = 0          # cache index of first token added this step
        self.step: int = 0

    # ---- bookkeeping -------------------------------------------------------
    def register(self, positions: Sequence[int], step: int, attn_received) -> None:
        self.new_start = len(self.positions)
        for p, a in zip(positions, attn_received):
            self.positions.append(int(p))
            self.entry_step.append(step)
            self._init_value(a)
        self.step = step

    def _init_value(self, attn) -> None:
        raise NotImplementedError

    def update_old(self, attn_old, step: int) -> None:
        """Default: no-op (only attention-accumulating policies use this)."""
        self.step = step

    def protect(self) -> List[int]:
        return list(range(self.new_start, len(self.positions)))

    # ---- decision ----------------------------------------------------------
    def choose_evict(self, n_evict: int) -> List[int]:
        n = len(self.positions)
        forbidden = set(self.protect())
        candidates = [i for i in range(n) if i not in forbidden]
        if n_evict >= len(candidates):
            return candidates[:n_evict]
        idx = self._rank_evict(candidates)
        return idx[:n_evict]

    def _rank_evict(self, candidates: List[int]) -> List[int]:
        raise NotImplementedError

    def reset(self) -> None:
        self.positions, self.entry_step = [], []
        self.new_start, self.step = 0, 0


class FullPolicy(EvictionPolicy):
    """(a) No eviction: upper-bound reference (capacity ignored)."""
    name = "full"

    def _init_value(self, attn) -> None:
        pass

    def _rank_evict(self, candidates):
        return []


class RandomPolicy(EvictionPolicy):
    """(b) Uniform random eviction among evictable tokens."""
    name = "random"

    def _init_value(self, attn) -> None:
        pass

    def _rank_evict(self, candidates):
        c = list(candidates)
        self.rng.shuffle(c)
        return c


class SlidingWindowPolicy(EvictionPolicy):
    """(c) Recency-only: drop the oldest non-protected tokens first."""
    name = "window"

    def _init_value(self, attn) -> None:
        pass

    def _rank_evict(self, candidates):
        return sorted(candidates, key=lambda i: self.positions[i])


class StreamingLLMPolicy(SlidingWindowPolicy):
    """(e) StreamingLLM: keep the first `sink_tokens` positions forever, drop the
    oldest of the rest. Matches Xiao et al. (2023) attention-sinks scheme."""
    name = "streaming"

    def __init__(self, capacity=None, seed=0, sink_tokens: int = 4):
        super().__init__(capacity, seed)
        self.sink_tokens = sink_tokens

    def protect(self):
        sinks = list(range(min(self.sink_tokens, len(self.positions))))
        return sinks + super().protect()


class H2OPolicy(EvictionPolicy):
    """(d) H2O: heavy-hitter oracle (Zhang et al. 2023). Score of a cached token is
    its *cumulative* attention received (plus tiny epsilon for ties); evict the
    lowest. Never evict tokens added in the current step (they are the recent
    window H2O keeps by construction of streaming generation)."""
    name = "h2o"

    def __init__(self, capacity=None, seed=0, eps: float = 1e-6):
        super().__init__(capacity, seed)
        self.cum_attn: List[float] = []
        self.eps = eps

    def _init_value(self, attn) -> None:
        self.cum_attn.append(float(attn) + self.eps)

    def update_old(self, attn_old, step: int) -> None:
        super().update_old(attn_old, step)
        for i, a in enumerate(attn_old):
            self.cum_attn[i] += float(a) + self.eps

    def _rank_evict(self, candidates):
        return sorted(candidates, key=lambda i: self.cum_attn[i])

    def reset(self) -> None:
        super().reset()
        self.cum_attn = []


class PESPolicy(EvictionPolicy):
    """(f) PES: Theorem 1's exact margin rule, Eq. (2).

    value  v_i   : EMA of attention received (running trace valuation)
    cost   h_i   : mu * (t - entry_step_i)  -- fast-store holding cost accrued
                   since the token entered the cache
    margin Pi_i : v_i - h_i
    Evict the lowest margin first; clip to capacity. Tokens whose margin has
    gone negative are (conceptually) routed to the slow store; since a dropped
    KV vector cannot be retrieved, the slow store here is 'dropped', so the
    only decision the capacity clip makes is *which* negative-margin tokens to
    keep while space remains.
    """
    name = "pes"

    def __init__(self, capacity=None, seed=0, ema_alpha: float = 0.1,
                 hold_cost_mu: float = 1e-3):
        super().__init__(capacity, seed)
        self.ema_alpha = ema_alpha
        self.mu = hold_cost_mu
        self.value: List[float] = []

    def _init_value(self, attn) -> None:
        self.value.append(float(attn))

    def update_old(self, attn_old, step: int) -> None:
        super().update_old(attn_old, step)
        a = self.ema_alpha
        for i, x in enumerate(attn_old):
            self.value[i] = (1 - a) * self.value[i] + a * float(x)

    def margin(self, i: int) -> float:
        """Pi(i, fast) per Eq. (2): v_i - mu * (t - entry_i)."""
        return self.value[i] - self.mu * (self.step - self.entry_step[i])

    def _rank_evict(self, candidates):
        return sorted(candidates, key=lambda i: self.margin(i))

    def reset(self) -> None:
        super().reset()
        self.value = []


def make_policy(name: str, capacity: Optional[int], seed: int = 0, **kw) -> EvictionPolicy:
    name = name.lower()
    if name == "full":
        return FullPolicy(None, seed)
    if name == "random":
        return RandomPolicy(capacity, seed, **kw)
    if name == "window":
        return SlidingWindowPolicy(capacity, seed, **kw)
    if name == "streaming":
        return StreamingLLMPolicy(capacity, seed, **kw)
    if name == "h2o":
        return H2OPolicy(capacity, seed, **kw)
    if name == "pes":
        return PESPolicy(capacity, seed, **kw)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICIES}")
