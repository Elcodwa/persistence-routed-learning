"""BudgetedKVCache: a fixed-token-capacity KV cache with pluggable eviction.

Subclasses transformers' DynamicCache so it plugs directly into any HF causal LM
via ``past_key_values=cache``. Token-level eviction decisions are made by an
EvictionPolicy (see policies.py) and applied uniformly across all layers.

Design notes
------------
* Eviction is performed *by the driver* (generate_with_cache) after each forward
  pass, using attention weights from that pass (attn_implementation="eager",
  output_attentions=True). This keeps the cache class simple and avoids
  version-specific internals: slicing is implemented by rebuilding the
  internal DynamicCache from the current tensors via update(), which is
  API-stable across transformers versions.
* Positions: we always pass explicit ``position_ids`` equal to each token's
  absolute original position. For RoPE models this is semantically correct
  after eviction; for learned-absolute models (GPT-2) it keeps the prompt
  within max_position_embeddings, so task contexts are kept < 1000 tokens.
* Prefill is chunked (chunk size configurable) so that eviction decisions
  during prefill are informed by real attention from the chunk being read.
  The most recent chunk is always protected from eviction.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformers import DynamicCache

from .policies import EvictionPolicy, make_policy


class BudgetedKVCache(DynamicCache):
    """DynamicCache with a token-capacity budget and an eviction policy."""

    def __init__(self, policy_name: str = "full", capacity: Optional[int] = None,
                 seed: int = 0, **policy_kw):
        super().__init__()
        self.policy: EvictionPolicy = make_policy(policy_name, capacity, seed=seed, **policy_kw)
        self.capacity = capacity if policy_name != "full" else None
        self.total_tokens = 0   # absolute count of tokens seen (== next position id)
        self.n_evictions = 0

    # ---- bookkeeping API used by the driver --------------------------------
    def observe(self, attn_matrix: torch.Tensor) -> None:
        """Register the tokens just added to the cache.

        attn_matrix: [n_new_query, cache_len] attention received (mean over
        heads) from this forward's queries to ALL cached tokens (old + new).
        The new tokens occupy the last n_new_query columns.
        """
        n_new = attn_matrix.shape[0]
        cache_len = attn_matrix.shape[1]
        n_old = cache_len - n_new
        positions = list(range(self.total_tokens, self.total_tokens + n_new))
        self.total_tokens += n_new
        # attention received by NEW tokens (their own columns, summed over queries)
        new_attn = attn_matrix[:, n_old:].sum(dim=0)
        self.policy.register(positions, self.policy.step + 1,
                             new_attn.detach().cpu().tolist())
        # attention received by OLD tokens from the new queries
        if n_old > 0:
            old_attn = attn_matrix[:, :n_old].sum(dim=0)
            self.policy.update_old(old_attn.detach().cpu().tolist(), self.policy.step)

    def evict_if_needed(self) -> int:
        """Clip cache to capacity per the policy. Returns number evicted."""
        if self.capacity is None:
            return 0
        seq_len = self.get_seq_length()
        n_evict = seq_len - self.capacity
        if n_evict <= 0:
            return 0
        idx = self.policy.choose_evict(n_evict)
        if not idx:
            return 0
        self._drop(idx)
        self.n_evictions += len(idx)
        return len(idx)

    def _drop(self, idx: List[int]) -> None:
        """Evict tokens at local indices `idx` from every layer (version-safe:
        rebuild the internal DynamicCache from sliced tensors)."""
        n = self.get_seq_length()
        keep = sorted(set(range(n)) - set(idx))
        keep_t = torch.tensor(keep, dtype=torch.long, device=self._device())
        new = DynamicCache()
        for li in range(len(self)):
            ko, vo = self._kv(li)
            new.update(ko.index_select(2, keep_t), vo.index_select(2, keep_t), li)
        self._swap(new)
        p = self.policy
        p.positions = [p.positions[i] for i in keep]
        p.entry_step = [p.entry_step[i] for i in keep]
        if hasattr(p, "cum_attn"):
            p.cum_attn = [p.cum_attn[i] for i in keep]
        if hasattr(p, "value"):
            p.value = [p.value[i] for i in keep]
        p.new_start = sum(1 for i in keep if i >= p.new_start)

    def _kv(self, layer_idx: int):
        if hasattr(self, "layers"):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _device(self):
        k, _ = self._kv(0)
        return k.device

    def _swap(self, other: DynamicCache) -> None:
        """Replace this cache's storage with `other`'s (used by _drop).

        Our own bookkeeping attrs are preserved; everything else (including
        DynamicCache internals like _seen_tokens) is taken from `other`.
        """
        ours = ("policy", "capacity", "total_tokens", "n_evictions", "question_len")
        keep_self = {k: self.__dict__[k] for k in ours if k in self.__dict__}
        self.__dict__.clear()
        self.__dict__.update(other.__dict__)
        self.__dict__.update(keep_self)


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------

def _attn_received(out, n_old: int) -> torch.Tensor:
    """Mean-over-heads attention of the last forward pass, shape [q, kv_len]."""
    # out.attentions: tuple over layers of [B, H, Q, KV]; use the LAST layer
    a = out.attentions[-1][0].float().mean(dim=0)   # [Q, KV]
    return a


@torch.no_grad()
def generate_with_cache(model, input_ids: torch.Tensor, cache: BudgetedKVCache,
                        max_new_tokens: int = 32, prefill_chunk: int = 64,
                        max_positions: int = 1024, eos_token_id: Optional[int] = None,
                        do_sample: bool = False, temperature: float = 1.0,
                        generator: Optional[torch.Generator] = None) -> dict:
    """Greedy/sample generation with attention-informed eviction.

    Prefill is processed in chunks so that mid-prefill eviction decisions are
    informed by real attention from the chunk being read (the newest chunk is
    always protected). The question is assumed to be the LAST chunk of
    `input_ids`; if `question_len` tokens should never be evicted, put them in
    the final chunk and pass `question_len > 0` so they stay protected.

    Returns dict with 'output_ids', 'n_evictions', 'peak_cache_len'.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    n_chunks = (prompt_len + prefill_chunk - 1) // prefill_chunk
    question_len = getattr(cache, "question_len", 0)

    cur = 0
    peak = 0
    while cur < prompt_len:
        end = min(cur + prefill_chunk, prompt_len)
        chunk = input_ids[:, cur:end]
        pos = torch.arange(cur, end, device=device)[None, :]
        out = model(chunk, position_ids=pos, past_key_values=cache,
                    output_attentions=True, use_cache=True)
        cache.observe(_attn_received(out, n_old=cur))
        # protect the question tail (the very last tokens of the prompt)
        if end >= prompt_len and question_len > 0:
            cache.policy.new_start = cache.get_seq_length() - question_len
        peak = max(peak, cache.get_seq_length())
        cache.evict_if_needed()
        cur = end

    # generation steps
    outputs = []
    next_tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    step = 0
    while step < max_new_tokens:
        tok = next_tok.item()
        if eos_token_id is not None and tok == eos_token_id and step > 0:
            break
        outputs.append(tok)
        if len(outputs) >= max_new_tokens:
            break
        pos = torch.tensor([[cache.total_tokens]], device=device)
        out = model(next_tok, position_ids=pos, past_key_values=cache,
                    output_attentions=True, use_cache=True)
        cache.observe(_attn_received(out, n_old=cache.get_seq_length() - 1))
        peak = max(peak, cache.get_seq_length())
        cache.evict_if_needed()
        logits = out.logits[:, -1]
        if do_sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1, generator=generator)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        step += 1

    return {"output_ids": torch.tensor(outputs, device=device),
            "n_evictions": cache.n_evictions, "peak_cache_len": peak}
