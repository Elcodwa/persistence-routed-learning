# Results (filled by Colab runs)

Status: **no model runs yet.** All infrastructure below is built and
unit-tested; the actual empirical numbers come from `notebooks/colab_runner.ipynb`
(T4 GPU). Run it, then `python kv_pes/scripts/make_report.py` regenerates this
file with real tables.

What will be reported here (all comparisons, favorable or not):

1. **Paired PES-vs-baseline tables** — per model × task × budget: mean, std,
   exact sign-test p, wins/losses/ties. Sign test mirrors the paper's own
   supplementary code.
2. **Dose-response curves** — accuracy vs cache budget (paper Fig. 4 analog).
3. **Needle-position curves** — early/mid/late retention under each policy.
4. **Boundary map** — where PES stops winning as distractor difficulty, cache
   tightness, model depth, and PES hyperparameters (mu, EMA alpha) vary; the
   paper's Sec. 7.4 predicts degradation under nonlinear credit assignment,
   and `delayed_needle` is the dedicated probe for exactly that failure mode.

If PES loses somewhere, it will be stated here in the same sentence-length
and font size as anywhere it wins.

## Pre-registered expectations (stated before any run, per the paper)

- PES > window/streaming when value ≈ cumulative attention (paper's central claim).
- PES vs H2O is the interesting pair: H2O's cumulative signal vs PES's EMA +
  holding-cost discount. Under `delayed_needle` (distractors after the needle),
  §7.4 predicts the EMA signal under-values the early needle → PES should
  degrade relative to H2O as distractor difficulty grows. If PES does NOT
  degrade there, that is a deviation from the paper's own failure prediction
  and will be reported as such.
