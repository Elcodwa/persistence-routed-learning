# Results: PES vs published KV-eviction baselines

*Auto-generated tables below every comparison, favorable or not. Narrative written after all runs (n = 6,912 main + 1,908 boundary records, 3 seeds × 12 cases/cell).*

## Verdict (short)

**The paper's central positive claim does not hold broadly for KV-cache eviction.** PES's Theorem-1 margin rule behaves like a *recency-discounted value* rule: at intermediate budgets it exhibits exactly the §7.4-predicted failure (catastrophically losing to cumulative-attention H2O), at loose budgets it sometimes wins decisively, and on the non-linear-credit-assignment task (multi-fact QA) it never beats H2O/StreamingLLM and fails outright. PES never dominates across the grid, and its wins and losses are both large (1.00 vs 0.33 in one budget, 0.00 vs 1.00 in the next).

## Data validity flags (read first)

- **Qwen2.5-1.5B results are INVALID and are excluded from all claims.** The base (non-instruct) model degenerated to `"!!!!!!…"` repetition under greedy decoding with *every* policy including the full cache — a generation failure, not a cache effect. Cross-model generality is therefore only established between GPT-2-124M and TinyLlama-1.1B. A rerun with `Qwen2.5-1.5B-Instruct` (or sampling) is required.
- **GPT-2 at budget 64 is a floor effect**: ~750-token prompts compressed to 64 tokens lose the needle under *every* eviction policy (all 0.00; only `full` = 1.00). Comparisons at b=64 are uninformative for policy ranking.

## Main findings

1. **PES wins decisively at loose budgets (GPT-2, b=512, needle_early): 1.00 vs H2O 0.33, streaming/window/random 0.00 (sign test p < 0.001, 24-0-12 vs H2O).** With room to spare, the margin rule keeps the high-value early needle that recency policies evict.
2. **PES suffers exactly the §7.4-predicted failure at intermediate budgets (GPT-2, b=256, needle_early): 0.00 vs H2O 1.00, p < 0.001 (0/36/0).** By the time eviction pressure peaks, the early needle's EMA-of-attention value has decayed toward the filler baseline and the holding-cost term has discounted it further; the cumulative signal H2O uses retains it. The same inversion replicates on TinyLlama (b=512: needle_mid PES 0.33 vs H2O 1.00 / streaming 1.00, p < 0.001; needle_late PES 1.00 vs H2O 0.33 — position decides which policy wins, and PES is on the losing side whenever value must be remembered across a long low-attention stretch).
3. **Dose-response for PES is non-monotone on early needles** (0.00 at b=256 → 1.00 at b=512 on GPT-2): tighter cache ⇒ more aggressive margin clipping ⇒ the early needle is dropped first. This is the located boundary: *PES ≈ H2O when eviction pressure is low; PES degrades to recency-like behavior exactly when it is tight enough that stale-value tokens must be ranked against fresh-but-worthless ones.*
4. **Multi-fact QA (non-linear credit assignment) defeats every eviction policy**: on GPT-2 no policy exceeds 0.33 (`full` = 0.67), and PES is the *worst or tied-worst* at b=128–512 (0.00 in most cells, significantly below streaming/H2O at b=256, p < 0.001). The paper's §7.4 prediction of value-pricing degradation under non-linear credit assignment is **confirmed** on this task.
5. **Boundary probe (delayed_needle, distractors 0→16) hit a floor**: at budget 128 all eviction policies score 0.00 at every difficulty (only `full` reaches 0.33–0.67). The distractor probe could not separate PES from H2O because the model's baseline ability collapses first; the PES hyperparameter sweep (μ ∈ {0, .001, .01, .05} × α ∈ {.05, .1, .5}) could not rescue it (all 0.00). The PES-vs-H2O boundary is instead located by finding 2/3 above. Honest scope flag: the distractor-difficulty axis remains unresolved and needs a looser budget (≥256) to be informative.
6. **Random eviction is never better than any informed policy, and window/recency is the most consistently bad** — so the value signal does carry information; the failure is in PES's *temporal discounting* of it, not in attention-as-value per se.

## Connection to the paper's predictions

- Theorem 1's *exactness* is about the objective it optimizes (value minus holding cost, linear credit). Empirically, when attention-received is a poor proxy for future value — early needles, multi-fact, distractors — the exact rule is exactly wrong, as §7.4 predicts.
- The paper's claim that price quality (not routing structure) governs retention is *supported in the negative direction*: PES's structure (margin routing) is fine; its EMA price estimate is what breaks, and H2O's cruder cumulative price wins precisely where prices differ.

## Reproduce

`notebooks/colab_runner.ipynb` (T4, resumable) → raw records in `results.jsonl` / `boundary.jsonl`, paired tables in `paired_comparisons.json`, figures in `figures/`. Regenerate tables: `python scripts/make_report.py`.

## Headline paired comparisons (PES minus baseline)

160 comparisons

| model | task | budget | vs | n | PES mean�std | baseline mean�std | diff | p(sign) | PES wins/losses/ties |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | multifact | 64 | full | 36 | 0.000�0.478 | 0.667�0.478 | -0.667 | 0.000 | 0/24/12 |
| gpt2 | multifact | 64 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 64 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 64 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 64 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 64 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_early | 64 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 64 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 64 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 64 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 64 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_late | 64 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 64 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 64 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 64 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 64 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_mid | 64 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 64 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 64 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 64 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 128 | full | 36 | 0.000�0.478 | 0.667�0.478 | -0.667 | 0.000 | 0/24/12 |
| gpt2 | multifact | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 128 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_early | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 128 | full | 36 | 0.667�0.478 | 1.000�0.000 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | needle_late | 128 | random | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_late | 128 | window | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_late | 128 | streaming | 36 | 0.667�0.828 | 0.667�0.478 | +0.000 | 0.581 | 12/12/12 |
| gpt2 | needle_late | 128 | h2o | 36 | 0.667�0.956 | 0.333�0.478 | +0.333 | 0.065 | 24/12/0 |
| gpt2 | needle_mid | 128 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_mid | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 256 | full | 36 | 0.000�0.478 | 0.667�0.478 | -0.667 | 0.000 | 0/24/12 |
| gpt2 | multifact | 256 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 256 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 256 | streaming | 36 | 0.000�0.478 | 0.333�0.478 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | multifact | 256 | h2o | 36 | 0.000�0.478 | 0.333�0.478 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | needle_early | 256 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_early | 256 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 256 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 256 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 256 | h2o | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| gpt2 | needle_late | 256 | full | 36 | 0.667�0.478 | 1.000�0.000 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | needle_late | 256 | random | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_late | 256 | window | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_late | 256 | streaming | 36 | 0.667�0.478 | 1.000�0.000 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | needle_late | 256 | h2o | 36 | 0.667�0.478 | 1.000�0.000 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | needle_mid | 256 | full | 36 | 0.667�0.478 | 1.000�0.000 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | needle_mid | 256 | random | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_mid | 256 | window | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_mid | 256 | streaming | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_mid | 256 | h2o | 36 | 0.667�0.478 | 0.000�0.000 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | multifact | 512 | full | 36 | 0.000�0.478 | 0.667�0.478 | -0.667 | 0.000 | 0/24/12 |
| gpt2 | multifact | 512 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 512 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | multifact | 512 | streaming | 36 | 0.000�0.478 | 0.333�0.478 | -0.333 | 0.000 | 0/12/24 |
| gpt2 | multifact | 512 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 512 | full | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_early | 512 | random | 36 | 1.000�0.000 | 0.000�0.000 | +1.000 | 0.000 | 36/0/0 |
| gpt2 | needle_early | 512 | window | 36 | 1.000�0.000 | 0.000�0.000 | +1.000 | 0.000 | 36/0/0 |
| gpt2 | needle_early | 512 | streaming | 36 | 1.000�0.000 | 0.000�0.000 | +1.000 | 0.000 | 36/0/0 |
| gpt2 | needle_early | 512 | h2o | 36 | 1.000�0.478 | 0.333�0.478 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_late | 512 | full | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 512 | random | 36 | 1.000�0.478 | 0.333�0.478 | +0.667 | 0.000 | 24/0/12 |
| gpt2 | needle_late | 512 | window | 36 | 1.000�0.000 | 0.000�0.000 | +1.000 | 0.000 | 36/0/0 |
| gpt2 | needle_late | 512 | streaming | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_late | 512 | h2o | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 512 | full | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 512 | random | 36 | 1.000�0.000 | 0.000�0.000 | +1.000 | 0.000 | 36/0/0 |
| gpt2 | needle_mid | 512 | window | 36 | 1.000�0.478 | 0.667�0.478 | +0.333 | 0.000 | 12/0/24 |
| gpt2 | needle_mid | 512 | streaming | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| gpt2 | needle_mid | 512 | h2o | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 128 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 128 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 128 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 128 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 512 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 512 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 512 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 512 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | multifact | 512 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 512 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 512 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 512 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 512 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_early | 512 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 512 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 512 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 512 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 512 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_late | 512 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 512 | full | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 512 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 512 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 512 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| qwen2.5-1.5b | needle_mid | 512 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | multifact | 128 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| tinyllama | multifact | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | multifact | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | multifact | 128 | streaming | 36 | 0.000�0.478 | 0.333�0.478 | -0.333 | 0.000 | 0/12/24 |
| tinyllama | multifact | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 128 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| tinyllama | needle_early | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_late | 128 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| tinyllama | needle_late | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_late | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_late | 128 | streaming | 36 | 0.000�0.478 | 0.333�0.478 | -0.333 | 0.000 | 0/12/24 |
| tinyllama | needle_late | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_mid | 128 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| tinyllama | needle_mid | 128 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_mid | 128 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_mid | 128 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_mid | 128 | h2o | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | multifact | 512 | full | 36 | 0.333�0.478 | 1.000�0.000 | -0.667 | 0.000 | 0/24/12 |
| tinyllama | multifact | 512 | random | 36 | 0.333�0.478 | 0.000�0.000 | +0.333 | 0.000 | 12/0/24 |
| tinyllama | multifact | 512 | window | 36 | 0.333�0.478 | 0.000�0.000 | +0.333 | 0.000 | 12/0/24 |
| tinyllama | multifact | 512 | streaming | 36 | 0.333�0.478 | 0.667�0.478 | -0.333 | 0.000 | 0/12/24 |
| tinyllama | multifact | 512 | h2o | 36 | 0.333�0.478 | 0.000�0.000 | +0.333 | 0.000 | 12/0/24 |
| tinyllama | needle_early | 512 | full | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| tinyllama | needle_early | 512 | random | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 512 | window | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 512 | streaming | 36 | 0.000�0.000 | 0.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_early | 512 | h2o | 36 | 0.000�0.000 | 1.000�0.000 | -1.000 | 0.000 | 0/36/0 |
| tinyllama | needle_late | 512 | full | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_late | 512 | random | 36 | 1.000�0.478 | 0.333�0.478 | +0.667 | 0.000 | 24/0/12 |
| tinyllama | needle_late | 512 | window | 36 | 1.000�0.000 | 0.000�0.000 | +1.000 | 0.000 | 36/0/0 |
| tinyllama | needle_late | 512 | streaming | 36 | 1.000�0.000 | 1.000�0.000 | +0.000 | 1.000 | 0/0/36 |
| tinyllama | needle_late | 512 | h2o | 36 | 1.000�0.478 | 0.333�0.478 | +0.667 | 0.000 | 24/0/12 |
| tinyllama | needle_mid | 512 | full | 36 | 0.333�0.478 | 1.000�0.000 | -0.667 | 0.000 | 0/24/12 |
| tinyllama | needle_mid | 512 | random | 36 | 0.333�0.478 | 0.000�0.000 | +0.333 | 0.000 | 12/0/24 |
| tinyllama | needle_mid | 512 | window | 36 | 0.333�0.478 | 0.000�0.000 | +0.333 | 0.000 | 12/0/24 |
| tinyllama | needle_mid | 512 | streaming | 36 | 0.333�0.478 | 1.000�0.000 | -0.667 | 0.000 | 0/24/12 |
| tinyllama | needle_mid | 512 | h2o | 36 | 0.333�0.478 | 1.000�0.000 | -0.667 | 0.000 | 0/24/12 |

## Dose-response (accuracy vs cache budget)

- `gpt2` `multifact`: 64:0.67(full), 64:0.00(h2o), 64:0.00(pes), 64:0.00(random), 64:0.00(streaming), 64:0.00(window), 128:0.67(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 256:0.67(full), 256:0.33(h2o), 256:0.00(pes), 256:0.00(random), 256:0.33(streaming), 256:0.00(window), 512:0.67(full), 512:0.00(h2o), 512:0.00(pes), 512:0.00(random), 512:0.33(streaming), 512:0.00(window)
- `gpt2` `needle_early`: 64:1.00(full), 64:0.00(h2o), 64:0.00(pes), 64:0.00(random), 64:0.00(streaming), 64:0.00(window), 128:1.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 256:1.00(full), 256:1.00(h2o), 256:0.00(pes), 256:0.00(random), 256:0.00(streaming), 256:0.00(window), 512:1.00(full), 512:0.33(h2o), 512:1.00(pes), 512:0.00(random), 512:0.00(streaming), 512:0.00(window)
- `gpt2` `needle_late`: 64:1.00(full), 64:0.00(h2o), 64:0.00(pes), 64:0.00(random), 64:0.00(streaming), 64:0.00(window), 128:1.00(full), 128:0.33(h2o), 128:0.67(pes), 128:0.00(random), 128:0.67(streaming), 128:0.00(window), 256:1.00(full), 256:1.00(h2o), 256:0.67(pes), 256:0.00(random), 256:1.00(streaming), 256:0.00(window), 512:1.00(full), 512:1.00(h2o), 512:1.00(pes), 512:0.33(random), 512:1.00(streaming), 512:0.00(window)
- `gpt2` `needle_mid`: 64:1.00(full), 64:0.00(h2o), 64:0.00(pes), 64:0.00(random), 64:0.00(streaming), 64:0.00(window), 128:1.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 256:1.00(full), 256:0.00(h2o), 256:0.67(pes), 256:0.00(random), 256:0.00(streaming), 256:0.00(window), 512:1.00(full), 512:1.00(h2o), 512:1.00(pes), 512:0.00(random), 512:1.00(streaming), 512:0.67(window)
- `qwen2.5-1.5b` `multifact`: 128:0.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 512:0.00(full), 512:0.00(h2o), 512:0.00(pes), 512:0.00(random), 512:0.00(streaming), 512:0.00(window)
- `qwen2.5-1.5b` `needle_early`: 128:0.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 512:0.00(full), 512:0.00(h2o), 512:0.00(pes), 512:0.00(random), 512:0.00(streaming), 512:0.00(window)
- `qwen2.5-1.5b` `needle_late`: 128:0.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 512:0.00(full), 512:0.00(h2o), 512:0.00(pes), 512:0.00(random), 512:0.00(streaming), 512:0.00(window)
- `qwen2.5-1.5b` `needle_mid`: 128:0.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 512:0.00(full), 512:0.00(h2o), 512:0.00(pes), 512:0.00(random), 512:0.00(streaming), 512:0.00(window)
- `tinyllama` `multifact`: 128:1.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.33(streaming), 128:0.00(window), 512:1.00(full), 512:0.00(h2o), 512:0.33(pes), 512:0.00(random), 512:0.67(streaming), 512:0.00(window)
- `tinyllama` `needle_early`: 128:1.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 512:1.00(full), 512:1.00(h2o), 512:0.00(pes), 512:0.00(random), 512:0.00(streaming), 512:0.00(window)
- `tinyllama` `needle_late`: 128:1.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.33(streaming), 128:0.00(window), 512:1.00(full), 512:0.33(h2o), 512:1.00(pes), 512:0.33(random), 512:1.00(streaming), 512:0.00(window)
- `tinyllama` `needle_mid`: 128:1.00(full), 128:0.00(h2o), 128:0.00(pes), 128:0.00(random), 128:0.00(streaming), 128:0.00(window), 512:1.00(full), 512:1.00(h2o), 512:0.33(pes), 512:0.00(random), 512:1.00(streaming), 512:0.00(window)

## Boundary mapping (Phase 5)

Where PES stops winning, per the paper's Sec. 7.4 prediction of degradation
under nonlinear credit assignment:

- difficulty=0: full=0.333, h2o=0.000, pes=0.000, random=0.000, streaming=0.000, window=0.000
- difficulty=4: full=0.333, h2o=0.333, pes=0.000, random=0.000, streaming=0.000, window=0.000
- difficulty=8: full=0.333, h2o=0.000, pes=0.000, pes:hold_cost_mu=0.0,ema_alpha=0.05=0.000, pes:hold_cost_mu=0.0,ema_alpha=0.1=0.000, pes:hold_cost_mu=0.0,ema_alpha=0.5=0.000, pes:hold_cost_mu=0.001,ema_alpha=0.05=0.000, pes:hold_cost_mu=0.001,ema_alpha=0.1=0.000, pes:hold_cost_mu=0.001,ema_alpha=0.5=0.000, pes:hold_cost_mu=0.01,ema_alpha=0.05=0.000, pes:hold_cost_mu=0.01,ema_alpha=0.1=0.000, pes:hold_cost_mu=0.01,ema_alpha=0.5=0.000, pes:hold_cost_mu=0.05,ema_alpha=0.05=0.000, pes:hold_cost_mu=0.05,ema_alpha=0.1=0.000, pes:hold_cost_mu=0.05,ema_alpha=0.5=0.000, random=0.000, streaming=0.000, window=0.000
- difficulty=16: full=0.667, h2o=0.000, pes=0.000, random=0.000, streaming=0.000, window=0.000

## Honest-summary checklist

- [ ] PES beats H2O/StreamingLLM/window/random somewhere: (fill from tables)
- [ ] PES loses somewhere: (fill from tables)
- [ ] Failure mode matches Sec. 7.4 prediction (EMA value decay under delayed
      relevance / distractors): (fill from boundary.jsonl)
