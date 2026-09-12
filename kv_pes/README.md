# PES-KV: An Empirical Test of "Priced Memory Allocation with Exact Trace Valuation" for LLM KV-Caches

This directory contains a **falsification test**, not a showcase. We take the
allocation rule proposed in *"Priced Memory Allocation with Exact Trace
Valuation: A Study on Linear Predictors"* — Persistence-Economics Selection
(PES): score each unit of information by value minus holding/moving cost,
route negative-margin items to slow storage, clip to capacity (Theorem 1) —
apply it to LLM KV-cache eviction (the paper's Table 4 mapping: trace = cached
K/V vectors, fast store = GPU memory, slow store = dropped, value = running
attention received), and measure whether it beats published eviction baselines
under a fixed memory budget.

**We also actively search for where it stops working.** The paper's own §7.4
predicts value-based pricing degrades under nonlinear credit assignment, and
attention is an instance of exactly that. A precise account of where the
theory's predictions hold and where they fail is the deliverable — reported
honestly either way.

## Methods

Policies (`src/kv_pes/policies.py`), all sharing one attention signal (attention
received per cached token from recent queries):

| id | policy | description |
|----|--------|-------------|
| `full` | no eviction | upper-bound reference |
| `random` | random | uniform eviction among cached tokens |
| `window` | sliding window | recency-only, drop oldest |
| `streaming` | StreamingLLM | attention sinks (first 4 tokens) + recent window |
| `h2o` | H2O | evict lowest *cumulative* attention received |
| `pes` | **PES** | Theorem 1 margin: `Pi_i = v_i - mu*(t - entry_i)` with `v_i` = EMA of attention received; evict lowest margin, clip to capacity |

Tasks (`src/kv_pes/tasks.py`): needle-in-a-haystack (needle at early/mid/late
depth), multi-fact long-context QA (8 facts at random depths; the paper's §7.4
stressor), and `delayed_needle` (early needle followed by attention-grabbing
distractor fake needles — a deliberate attempt to break EMA-based valuation).

All sweeps checkpoint every generation to JSONL (`results/results.jsonl`) and
resume across Colab sessions. Statistics are paired exact sign tests on
per-case differences (matching the paper's own supplementary code), with means,
stds, and p-values reported for **every** comparison, not only favorable ones.

## Repo layout

```
kv_pes/
  src/kv_pes/   cache.py (BudgetedKVCache, Cache-API subclass) · policies.py ·
                tasks.py · runner.py · stats.py · analysis.py · plots.py
  tests/        unit tests for eviction logic, cache invariants, tasks, stats
  notebooks/    colab_runner.ipynb  <- single entry point for all real runs (T4)
  scripts/      run_sweep.py (CLI: model, policies, budgets, seeds, difficulty)
  results/      raw JSONL + paired_comparisons.json + figures (filled by runs)
```

## How to run

Open `notebooks/colab_runner.ipynb` in Google Colab with a T4 runtime. It:
1. clones the repo and installs the package,
2. runs the unit-test gate (must pass before any real evaluation),
3. runs a 2-minute smoke test,
4. runs the Phase 3 sweep (GPT-2-124M full grid: 6 policies × 4 budgets ×
   3 needle depths + multi-fact × 3 seeds, n=12 cases/cell; then TinyLlama-1.1B
   and Qwen2.5-1.5B subsets),
5. runs the Phase 5 boundary sweep (distractor difficulty 0→16, PES
   hyperparameter grid `mu × alpha`, plus a larger-model probe),
6. produces paired-comparison tables and dose-response / needle-position /
   boundary plots.

Estimated T4 time: GPT-2 grid ≈ 2–3 h; larger-model subsets ≈ 4–6 h;
boundary sweeps ≈ 2–3 h. Run across multiple sessions if needed — nothing is lost.

## Results

To be filled from the runs (raw records + `results.md` summary). The summary
must report both positive and negative findings relative to the paper's claims,
including any observed failure of the PES margin rule.

## Status

- [x] Phase 1: cache + 6 policies + unit tests
- [x] Phase 2: tasks + grading (n = 12/cell)
- [x] Phase 3/4/5 infrastructure: resumable sweeps, paired stats, boundary knobs
- [x] Phase 6: plotting + reporting scaffolding
- [ ] Real model sweeps (Colab T4) — run `notebooks/colab_runner.ipynb`
- [ ] results.md from actual runs
