# FAILED IDEAS LEDGER

Format: hypothesis → why promising → experiment/probe → result → failure explanation → lesson →
does failure suggest a better hypothesis?

## F-01. "Learned write-gating into a fast state" as the core contribution [REJECTED as insufficient]
- Hypothesis: letting a network learn WHEN to write into a decaying fast memory is the missing principle.
- Why promising: Titans/GatedDeltaNet showed big gains from gated writes.
- Probe: literature wave 3 (DeltaNet, Gated Delta Networks, TTT-KV-binding, DNC lineage).
- Result: per-item learned write/erase gates ALREADY EXIST widely (DeltaNet gates, DNC allocation/free-list,
  sparse access memory write priorities).
- Failure explanation: single-store write gating is a solved sub-problem; claiming it would be renaming.
- Lesson: the defensible object is NOT the write gate but CROSS-STORE ROUTING under an economic objective
  spanning activation→moment→parameter media with maintenance costs and derived special cases.
- Better hypothesis: yes → H01 (routing/economics), with H16 as deployment regime.
- Status: absorbed into H01.

## F-02. "Nested Learning already did it" [CHECK — resolved as NOT blocking, but constraining claims]
- Probe: full abstract + blog of 2512.24695.
- Result: NL establishes nested optimizers-as-memory with DESIGNED levels/context flows; no derivation of
  level structure, no priced routing, no maintenance economics, no special-case recovery theorem.
- Lesson: position H01 explicitly as answering "WHICH memories deserve WHICH persistence" — the question NL
  leaves open; cite NL as the nearest framework, never claim NL-style nesting itself.
- Status: constraint recorded; H01 survives provisionally.

## F-03. "Consolidation = compression (IB)" as the core principle [REJECTED as insufficient]
- Probe: read 2603.04688 (predictive forgetting).
- Result: normative compression account of consolidation exists for latent representations (offline).
- Lesson: extend, don't re-derive: our delta must be ONLINE, PARAMETER-STACK-WIDE, DECISION-THEORETIC
  (routing with costs), not another compression story.
- Status: absorbed as theory component of H01.

## F-04. H51 Superposition-hypothesis stores [REJECTED early]
- Probe: internal critique.
- Result: mechanism is Hopfield-mixture algebra renamed; interference-as-feature claim not clearly
  separable from dense associative memory behavior; low feasibility score.
- Lesson: killed at generation stage; recorded for honesty.

## F-05. H06 Store-topology NAS, H49 Landauer-priced loss, H50 monoidal learners [DEFERRED — not failed]
- Reason: interesting but either infeasible in tiny experiments (H49 needs physical grounding, H50 needs
  category-theoretic machinery with weak empirical hook) or likely to produce engineering not principle (H06).
- Status: parked in rejected_ideas as deferred; revisit only if H01 collapses.

## F-06. Utility-density valuation u = v/(eps+|w|) [REJECTED by falsification]
- Hypothesis: pricing traces by value PER UNIT WEIGHT (cost ∝ norm) implements interference pricing.
- Experiment: unit tests + stationary-env sanity runs (pes_tests.py).
- Result: catastrophic — zero-weight newcomers have vanishing denominator => maximal utility density;
  router evicts TRAINED slow traces to admit IGNORANT arrivals; loss rises monotonically after fit.
- Failure explanation: norm-cost punishes exactly what learning produces; value and cost must be
  independent coordinates (tariffs λ, μ, κ), not ratios.
- Lesson: kept as a documented negative result — naive economic intuitions can invert under coupling.
- Better hypothesis: additive tariffs with raw responsibility value (current design).
- Status: fixed in code; recorded for paper's ablation narrative.

## F-09. "Hysteresis suppresses spike-locking" retrodiction [REJECTED by its own confirmation test]
- Hypothesis: P3 null at default settings exists because switching costs throttle reactivity;
  relaxing hysteresis should restore error-coupled promotions.
- Experiment: followup_probes.py R2 — hysteresis ∈ {1,5,10}, contrast err(lag5)/err(lag30).
- Result: contrast ≈ 1.02 / 1.00 / 1.00 — NO re-emergence. Retrodiction falsified.
- Better hypothesis: the value EMA (ρ=0.98 ⇒ ~50-step integration window) makes promotions respond to
  INTEGRATED responsibility, not instantaneous spikes; the router's own memory timescale governs its
  reactivity. Registered prediction: lowering ρ should produce spike-locking (see rho_probe.json).
- Lesson: two mechanisms can produce the same null; only a differential prediction separates them.
- Status: superseded by ρ-integration account.

## F-10. "Value-EMA integration window explains absent spike-locking" [REJECTED]
- Hypothesis: promotions respond to integrated responsibility over the ρ-window, so fast EMA (low ρ)
  should create spike-locking.
- Experiment: rho_probe.json — ρ ∈ {0.9, 0.98, 0.998}, contrast err(lag5)/err(lag30).
- Result: contrast ≈ 0.996/1.000/0.994 — flat. Rejected.
- Root cause discovered: promotion count IDENTICAL (765) across all ρ ⇒ routing operates at
  SWITCH-BUDGET SATURATION (one swap per hysteresis opening); timing carries no value information,
  so no coupling mechanism can manifest regardless of valuation dynamics.
- Emergent synthesis (recorded as finding, not failure): the routing law has THREE operating regimes —
  (i) saturated: timing = switching budget (our volatile env here), (ii) selective: timing = value
  threshold crossings (where error-coupling/P3 becomes testable; requires larger κ so most openings
  decline), (iii) frozen: below tariff floor (dose-response plateau). Prior probes sampled regimes
  (i)/(iii); the P3 question remains open ONLY in regime (ii) — registered as R2'.
- Lesson: before measuring *when* a controller acts, verify its action budget isn't saturated;
  otherwise you measure the clock, not the policy.
- Status: closed; R2' registered in RESULTS_NARRATIVE.md.

## F-11. "Capacity pressure re-opens PES-vs-clock gap on MLP" [REJECTED]
- Hypothesis (registered after Phase 6): greedy-exchange advantage re-emerges as caps → support size.
- Experiment: r6_capacity.py — caps ∈ {14+6, 20+10, 30+15} × {stationary, drifting}, n=3 paired seeds.
- Result: OPPOSITE. Pressure amplifies clock's advantage (14+6 stationary: clock 0.0127 vs pes 0.1004);
  pes edges drift only at the loosest setting (30+15: 0.1497 vs 0.1510).
- Diagnosis: under pressure, routing frequency rises exactly where the first-order value signal
  (∂ŷ/∂x_i · x_i, EMA-smoothed) is least reliable — through a tanh layer it misses interaction terms,
  so pressured exchanges evict valuable traces based on misattributed credit. Fixed-clock consolidation
  swaps rarely and ranks by accumulated value only at swap moments, which filters noise.
- Lesson: pricing requires a trustworthy price; naive gradient-derived prices degrade precisely when
  the market gets busy. Upgrading the valuation functional (leave-one-out / influence estimates /
  learned critic / longer integration) is THE open problem for making memory economics competitive on
  nonlinear substrates.
- Status: closed; scope statement updated everywhere.

---
Running tally: 4 rejected/killed outright, 3 deferred, 45+ still alive in ledger at various scores.
