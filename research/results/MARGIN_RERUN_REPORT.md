# MARGIN-RULE RE-RUN (2026-08-26, post-Proposition-correction) — FULL HONESTY REPORT

## What happened
Numerical verification of the propositions exposed a statement/implementation mismatch:
proofs described a margin rule (rank by Pi(i,slow)-Pi(i,fast)); the code ran a level rule
(argmax Pi(i,slow)). Margin rule implemented; first attempt produced a churn loop
(promote-evict-reenter cycle every 25 steps); root cause: slow-trace staleness never advanced.
Fixed by refreshing last_touch for all resident coordinates each step. All 23 unit tests green.

## Corrected-rule results (n=10, T=3000, caps 45+15)
### Bernoulli family (original)
| regime | pes | clock | single | random | notes |
|---|---|---|---|---|---|
| stationary | 0.161±0.058 | **0.089±0.042** | 0.154±0.060 | 0.229±0.098 | clock best; random sig. worse than PES (p=.021) |
| drifting | 0.173±0.059 | 0.164±0.049 | 0.161±0.043 | 0.196±0.070 | all valued policies tie |
| volatile | 0.280±0.060 | 0.292±0.053 | 0.288±0.066 | 0.256±0.048 | no separation |

### Gaussian family
All four methods statistically indistinguishable in all three regimes (e.g., drift: 0.345 vs 0.348 vs 0.349 vs 0.348).

## Interpretation (what changed vs level-rule results)
1. The margin rule is a STRICTER router (value must clear placement DIFFERENCES), so it promotes less
   aggressively; on stationary Bernoulli it now loses to clock — consistent with R6's finding that more
   routing decisions through noisy prices hurt retention-dominated regimes.
2. The earlier "drift dominance" claims DO NOT survive the corrected rule at these scales. The
   value-signal NECESSITY result partially survives (random still significantly worse in stationary).
3. The Gaussian family shows flat behavior for all policies — dense activations + margin strictness =
   little differentiation.

## Where this leaves the paper's empirical section
The strong per-regime dominance claims written earlier (based on the level rule) are WITHDRAWN.
What remains supported across BOTH rules:
- Value-signal necessity under capacity pressure (random worst or tied-worst everywhere).
- Transition-cost dose-response law and three-regime structure (routing-frequency phenomena,
  independent of donor-selection rule).
- Ablation ordering (switching cost/hysteresis/decay/tariff each contribute).
- The MLP non-replication and price-trust diagnosis (unchanged — different substrate).
New honest headline: "the economics framework organizes the design space; naive instantiations of the
router do not yet beat simple schedules; valuation quality is the binding constraint."

## Decision recorded
Paper §5 rewritten around the corrected numbers; dominance language removed; framework+negative-results
framing strengthened. This is the version that matches the code that will be published.
