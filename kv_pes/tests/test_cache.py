"""Unit tests for BudgetedKVCache (Phase 1 correctness gate).

Uses small synthetic tensors -- no model required -- plus (optionally, if
torch is available with transformers) a tiny randomly-initialized GPT-2 built
in memory for an end-to-end smoke test of generate_with_cache. The smoke test
is skipped automatically if transformers is missing; it downloads nothing.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kv_pes.cache import BudgetedKVCache
from kv_pes.policies import SlidingWindowPolicy

L, B, H, D = 2, 1, 2, 4


def fake_kv(n, device="cpu", dtype=torch.float32):
    k = torch.randn(B, H, n, D, device=device, dtype=dtype)
    v = torch.randn(B, H, n, D, device=device, dtype=dtype)
    return k, v


def fill(cache, n, step=1):
    """Add n tokens' KV across all layers + register them in the policy."""
    for li in range(L):
        k, v = fake_kv(n)
        cache.update(k, v, li)
    cache.policy.register(range(cache.total_tokens, cache.total_tokens + n),
                          step, [0.1] * n)
    cache.total_tokens += n


def test_capacity_enforced():
    c = BudgetedKVCache("window", capacity=6)
    fill(c, 10)
    n = c.evict_if_needed()
    assert c.get_seq_length() == 6
    assert n == 4
    assert len(c.policy.positions) == 6


def test_all_layers_evicted_consistently():
    c = BudgetedKVCache("random", capacity=5, seed=3)
    fill(c, 8)
    c.evict_if_needed()
    assert all(c._kv(li)[0].shape[2] == 5 for li in range(L))
    assert len(c.policy.positions) == 5


def test_window_evicts_oldest_positions():
    c = BudgetedKVCache("window", capacity=4)
    fill(c, 6)
    c.evict_if_needed()
    assert c.policy.positions == [2, 3, 4, 5]


def test_full_ignores_capacity():
    c = BudgetedKVCache("full", capacity=2)
    fill(c, 10)
    c.evict_if_needed()
    assert c.get_seq_length() == 10


def test_policy_state_stays_aligned_after_eviction():
    c = BudgetedKVCache("h2o", capacity=4)
    fill(c, 6, step=1)
    c.evict_if_needed()
    assert len(c.policy.cum_attn) == c.get_seq_length()
    # PES variant too
    c2 = BudgetedKVCache("pes", capacity=4)
    fill(c2, 6, step=1)
    c2.evict_if_needed()
    assert len(c2.policy.value) == c2.get_seq_length()
    assert len(c2.policy.entry_step) == c2.get_seq_length()


def test_streaming_sinks_survive_eviction():
    c = BudgetedKVCache("streaming", capacity=5, sink_tokens=2)
    fill(c, 9)
    c.evict_if_needed()
    assert 0 in c.policy.positions and 1 in c.policy.positions
    assert c.get_seq_length() == 5


def _tiny_gpt2():
    """Randomly-initialized tiny GPT-2 built in memory (no download)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=128, n_positions=128, n_embd=32, n_layer=2,
                     n_head=2)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    model.config._attn_implementation = "eager"
    return model


def _tiny_tokenizer():
    """Whitespace char tokenizer stub so tasks run without downloads."""
    class Tok:
        pad_token = "<pad>"
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=True, **kw):
            ids = [(ord(ch) % 128) for ch in text][:120]
            return {"input_ids": ids}
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(i) % 128) for i in ids)
    return Tok()


def test_generate_with_cache_smoke():
    try:
        from transformers import __version__ as tf_version  # noqa
    except ImportError:
        pytest.skip("transformers not installed")
    from kv_pes.cache import generate_with_cache
    model = _tiny_gpt2()
    tok = _tiny_tokenizer()
    ids = torch.tensor([[(i * 7 + 3) % 120 for i in range(100)]])
    for policy, cap in [("full", None), ("window", 32), ("h2o", 32),
                        ("streaming", 32), ("pes", 32), ("random", 32)]:
        cache = BudgetedKVCache(policy, cap, seed=0)
        cache.question_len = 10
        out = generate_with_cache(model, ids, cache, max_new_tokens=5,
                                  prefill_chunk=40, max_positions=128)
        assert out["output_ids"].numel() == 5, (policy, out)
        if cap is not None:
            assert out["peak_cache_len"] <= cap + 40, (policy, out["peak_cache_len"])
            assert out["n_evictions"] > 0
