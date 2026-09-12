"""PES-KV: empirical test of Persistence-Economics Selection (PES) for LLM KV-cache eviction.

Implements the paper's Theorem 1 margin rule (value minus holding cost, clip to capacity)
against published baselines (H2O, StreamingLLM, sliding window, random) on long-context tasks.
"""
__version__ = "0.1.0"

from .policies import (
    EvictionPolicy,
    FullPolicy,
    RandomPolicy,
    SlidingWindowPolicy,
    StreamingLLMPolicy,
    H2OPolicy,
    PESPolicy,
    make_policy,
    POLICIES,
)
from .cache import BudgetedKVCache, generate_with_cache

__all__ = [
    "EvictionPolicy", "FullPolicy", "RandomPolicy", "SlidingWindowPolicy",
    "StreamingLLMPolicy", "H2OPolicy", "PESPolicy", "make_policy", "POLICIES",
    "BudgetedKVCache", "generate_with_cache",
]
