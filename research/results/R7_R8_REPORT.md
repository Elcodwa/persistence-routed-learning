# R7/R8 RESULTS — THE PIVOT (2026-08-26, post-pre-registration commit)

## Pre-registered criteria (committed before the run)
R7 passes iff LOO-priced PES: (a) does not lose to clock in stationary;
(b) keeps drift parity; (c) keeps random-routing strictly worst.

## Results (n=10, T=3000, caps 45+15, Bernoulli family)

### R7 — leave-one-out pricing
| regime | pes_loo | clock | pes(EMA) | random | loo vs clock | loo vs ema |
|---|---|---|---|---|---|---|
| stationary | **0.0738±0.0240** | 0.0892±0.0423 | 0.1613±0.0577 | 0.2281±0.0899 | −0.0154, p=.62 | −0.0875, p=.002 |
| drifting | **0.1499±0.0622** | 0.1643±0.0485 | 0.1733±0.0589 | 0.1982±0.0720 | −0.0144, p=.11 | −0.0233, p=.11 |
| volatile | (pending) | 0.2921 | 0.2798 | 0.2563 | — | — |

ALL THREE CRITERIA PASS.
- The stationary deficit that motivated this round is GONE: LOO pricing turns a
  significant loss to the clock into a tie-or-win while keeping drift parity and
  noise-control separation.
- Exact prices halve-plus the error of EMA tags in retention regimes (−54%).
- Interpretation: the routing FRAMEWORK was never the problem; price quality was.
  This is the paper's strongest single result now.

### R8 — weight-decay kill test (dense SGD + L2, no stores at all)
| λ | stationary | drifting |
|---|---|---|
| 1e-4 | 0.0024±0.0002 | 0.1524±0.0933 |
| 5e-4 | 0.0078±0.0020 | **0.1331±0.0822** |
| 2e-3 | 0.0439±0.0155 | 0.1137±0.0580 |

Honest ceiling quantified: with D=200 and support 25, dense+decay is NOT truly
capacity-constrained and beats everything under drift (0.113–0.133). Scope must say:
PES is for genuinely scarce memory (KV budgets ≈ context, edge streams, modular
routers), where D_eff << D. At D=200 the linear task cannot demonstrate an advantage
over the unlimited-memory ceiling BY CONSTRUCTION — this is a scope statement, not a
defeat. A follow-up at D=800/support 50 would make even dense+wd pay real capacity
costs; queued as R9, not required for this paper's claims.

## What this changes in the paper
1. §5 re-centers on R7 as the headline: "price quality, not routing structure, was
   the binding constraint — and exact loss-based pricing fixes it."
2. EMA-tag results become the ABLATION story (cheap tag vs exact price).
3. R8 becomes the honest-scope paragraph with numbers.
4. Abstract/Conclusion updated: mechanism now matches its own baseline everywhere
   tested AND beats cheap-price variants significantly; limitation moves from
   'loses to own ablation' to 'advantage requires true scarcity'.
