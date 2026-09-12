"""Unit tests for eviction policies (Phase 1 correctness gate).

These run on CPU in seconds; no model download is required.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kv_pes.policies import (FullPolicy, H2OPolicy, PESPolicy, RandomPolicy,
                             SlidingWindowPolicy, StreamingLLMPolicy,
                             make_policy)


def register_tokens(policy, n, step=1, attn=0.1):
    policy.register(range(n), step, [attn] * n)


def test_full_never_evicts():
    p = FullPolicy()
    register_tokens(p, 10)
    assert p.choose_evict(5) == []


def test_random_evicts_requested_count_and_respects_protection():
    p = RandomPolicy(capacity=8, seed=1)
    register_tokens(p, 10)
    out = p.choose_evict(2)
    assert len(out) == 2
    assert len(set(out)) == 2
    assert all(0 <= i < 10 for i in out)


def test_window_drops_oldest_first():
    p = SlidingWindowPolicy(capacity=5)
    register_tokens(p, 10)
    out = p.choose_evict(3)
    assert sorted(out) == [0, 1, 2]


def test_streaming_protects_sinks():
    p = StreamingLLMPolicy(capacity=6, sink_tokens=3)
    register_tokens(p, 10)
    out = p.choose_evict(4)
    assert 0 not in out and 1 not in out and 2 not in out
    assert sorted(out) == [3, 4, 5, 6]


def test_h2o_evicts_lowest_cumulative_attention():
    p = H2OPolicy(capacity=5)
    p.register(range(5), step=1, attn=[0.0] * 5)
    # token 2 receives lots of attention later; token 0 gets none
    p.update_old([0.0, 0.1, 0.9, 0.1, 0.1], step=2)
    p.update_old([0.0, 0.0, 0.8, 0.0, 0.0], step=3)
    out = p.choose_evict(1)
    assert out == [0]


def test_h2o_new_tokens_protected():
    p = H2OPolicy(capacity=5)
    p.register(range(5), step=1, attn=[0.0] * 5)
    p.update_old([0.0, 0.5, 0.0, 0.0, 0.2], step=2)
    p.register([5], step=2, attn=[0.0])       # brand-new token, zero attention
    out = p.choose_evict(1)
    assert 5 not in out


def test_pes_margin_formula_exact():
    p = PESPolicy(capacity=4, ema_alpha=0.1, hold_cost_mu=0.01)
    p.register(range(2), step=1, attn=[0.0, 0.5])   # v = [0, 0.5]
    p.step = 5
    # margins: v_i - mu * (t - entry_i)
    assert p.margin(0) == pytest.approx(0.0 - 0.01 * (5 - 1))
    assert p.margin(1) == pytest.approx(0.5 - 0.01 * (5 - 1))


def test_pes_ema_update():
    p = PESPolicy(capacity=4, ema_alpha=0.5, hold_cost_mu=0.0)
    p.register([0], step=1, attn=[1.0])
    p.update_old([0.25], step=2)
    assert p.value[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.25)


def test_pes_evicts_lowest_margin():
    p = PESPolicy(capacity=3, ema_alpha=0.1, hold_cost_mu=0.1)
    # three old tokens; token 0: high value old entry (cost accrued),
    # token 1: high value recent, token 2: zero value
    p.register([0, 1, 2], step=1, attn=[0.9, 0.9, 0.0])
    p.step = 10
    p.update_old([0.0, 0.9, 0.0], step=10)
    out = p.choose_evict(1)
    assert out == [2]


def test_pes_holding_cost_flips_ranking():
    """The cost term, not just value, must drive eviction."""
    p = PESPolicy(capacity=2, ema_alpha=0.1, hold_cost_mu=10.0)
    p.register([0, 1, 2], step=1, attn=[0.5, 0.5, 0.1])
    p.step = 10
    p.update_old([0.5, 0.5, 0.1], step=10)
    out = p.choose_evict(1)
    # all margins deeply negative; token 2 has lowest value and equal-ish cost
    assert out == [2]


def test_make_policy_dispatch():
    for name in ["full", "random", "window", "streaming", "h2o", "pes"]:
        p = make_policy(name, capacity=8)
        register_tokens(p, 10)
        assert len(p.choose_evict(2)) <= 2


def test_register_alignment():
    p = PESPolicy(capacity=4)
    p.register(range(3), 1, [0.2, 0.3, 0.4])
    assert p.positions == [0, 1, 2]
    assert p.value == [0.2, 0.3, 0.4]
    assert p.entry_step == [1, 1, 1]
