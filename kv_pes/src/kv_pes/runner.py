"""Experiment runner: policies x tasks x budgets x seeds, with JSONL checkpointing.

Every completed (case, policy) generation is appended to results.jsonl keyed by
(model, task, difficulty, budget, policy, seed, case_id); re-running the same
sweep resumes by skipping keys already present. This is the Colab survivability
mechanism: a T4 session that dies mid-sweep loses at most one record.

Usage: see scripts/run_sweep.py and notebooks/colab_runner.ipynb.
"""
from __future__ import annotations

import json
import os
import random
import time
import zlib
from typing import List, Optional

import torch

from .cache import BudgetedKVCache, generate_with_cache
from .policies import POLICIES
from .tasks import (NeedleCase, grade_answer, grade_multi, multi_fact_qa,
                    needle_in_haystack, tokenize_case)

MODELS = {
    "gpt2": {"hf_id": "gpt2", "max_positions": 1024, "dtype": "float32"},
    "tinyllama": {"hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "max_positions": 2048, "dtype": "float16"},
    "qwen2.5-1.5b": {"hf_id": "Qwen/Qwen2.5-1.5B", "max_positions": 4096, "dtype": "float16"},
}


def build_case(task: str, seed: int, difficulty: int = 0, n_filler_words: int = 600) -> NeedleCase:
    """Deterministically build one eval case; depth is jittered around the
    nominal anchor to avoid degenerate identical prompts across seeds."""
    rng = random.Random(10_000 * seed + zlib.crc32(task.encode()) % 9973 + difficulty * 31)
    if task.startswith("needle_early"):
        return needle_in_haystack(rng, depth=0.10 + rng.random() * 0.05,
                                  n_filler_words=n_filler_words, distractors=difficulty)
    if task.startswith("needle_mid"):
        return needle_in_haystack(rng, depth=0.48 + rng.random() * 0.05,
                                  n_filler_words=n_filler_words, distractors=difficulty)
    if task.startswith("needle_late"):
        return needle_in_haystack(rng, depth=0.85 + rng.random() * 0.05,
                                  n_filler_words=n_filler_words, distractors=difficulty)
    if task == "delayed_needle":
        # needle early, then attention-grabbing distractor fake needles after it
        return needle_in_haystack(rng, depth=0.05, n_filler_words=n_filler_words,
                                  distractors=max(4, difficulty))
    if task == "multifact":
        return multi_fact_qa(rng, k=8, n_filler_words=n_filler_words)


def run_one(model, tokenizer, case: NeedleCase, policy: str, capacity: Optional[int],
            seed: int, device: str, prefill_chunk: int = 64, max_new_tokens: int = 16,
            pes_kw: Optional[dict] = None, task: str = "") -> dict:
    """Run one generation under one policy; returns a JSON-safe result record."""
    pes_kw = pes_kw or {}
    cache = BudgetedKVCache(policy, capacity, seed=seed, **pes_kw)
    cache.question_len = case.question_len
    ids = torch.tensor([case.meta["prompt_token_ids"]], device=device)
    out = generate_with_cache(model, ids, cache, max_new_tokens=max_new_tokens,
                              prefill_chunk=prefill_chunk, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out["output_ids"], skip_special_tokens=True)
    score = grade_multi(text, case.answer) if ":" in case.answer else float(grade_answer(text, case.answer))
    return {
        "score": score, "gen_text": text[:200],
        "n_evictions": out["n_evictions"], "peak_cache_len": out["peak_cache_len"],
        "n_prompt_tokens": len(ids[0]), "capacity": capacity, "policy": policy,
        "answer": case.answer, "needle_pos": case.needle_char_pos,
        "key": case.key, "task": task,
    }


class ResultStore:
    """Append-only JSONL store with resume support."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    self.done.add(self._key(json.loads(line)))
        self._fh = open(path, "a")

    @staticmethod
    def _key(r: dict) -> tuple:
        return (r["model"], r["task"], r["difficulty"], r["budget"],
                r["policy"], r["seed"], r["case_id"])

    def add(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.done.add(self._key(record))



def sweep(model_cfg_name: str, model, tokenizer, policies: List[str],
          budgets: List[Optional[int]], tasks: List[str], seeds: List[int],
          n_cases_per_cell: int = 12, difficulty: int = 0,
          store: Optional[ResultStore] = None, device: str = "cpu",
          prefill_chunk: int = 64, pes_grid: Optional[List[dict]] = None,
          log=print) -> List[dict]:
    """Main loop. `pes_grid` allows extra PES hyperparameter variants (Phase 5
    boundary mapping), e.g. [{'hold_cost_mu': 0.01}, {'ema_alpha': 0.5}]."""
    case_cache: dict = {}

    def get_case(task, seed, d):
        ck = (task, seed, d)
        if ck not in case_cache:
            case_cache[ck] = tokenize_case(build_case(task, seed, d), tokenizer)
        return case_cache[ck]

    records = []
    for policy in policies:
        pes_variants = [{}] if policy != "pes" else (pes_grid or [{}])
        for pes_kw in pes_variants:
            ptag = policy if not pes_kw else \
                f"{policy}:" + ",".join(f"{k}={v}" for k, v in pes_kw.items())
            for budget in budgets:
                for task in tasks:
                    for seed in seeds:
                        for ci in range(n_cases_per_cell):
                            case = get_case(task, seed, difficulty)
                            case.case_id = f"{task}|d{difficulty}|s{seed}|c{ci}"
                            rec_key = {"model": model_cfg_name, "task": task,
                                       "difficulty": difficulty, "budget": budget or -1,
                                       "policy": ptag, "seed": seed, "case_id": case.case_id}
                            if store is not None and store._key(rec_key) in store.done:
                                continue
                            t0 = time.time()
                            r = run_one(model, tokenizer, case, policy.split(":")[0],
                                        budget, seed, device, prefill_chunk,
                                        pes_kw=pes_kw, task=task)
                            r.update(rec_key)
                            r["runtime_s"] = round(time.time() - t0, 2)
                            records.append(r)
                            if store is not None:
                                store.add(r)
                            log(f"{model_cfg_name} {ptag} cap={budget} {task} s{seed} c{ci} "
                                f"score={r['score']:.0f} ev={r['n_evictions']} ({r['runtime_s']}s)")
    return records

