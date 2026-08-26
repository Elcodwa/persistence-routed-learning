# PES Research Code — Reproduction Guide

## What this is
Minimal experimental testbed for **Persistence-Economics Selection (PES)**: a learning principle where
the routing of information between memory stores (fast/decaying vs slow/protected) is *derived online*
from one priced objective (trace value minus maintenance tariffs minus switching costs), rather than
being fixed by architecture or schedule.

## Layout
- `pes_core.py`      environment + learners (all share predictor & SGD; differ ONLY in store policy)
- `pes_tests.py`     23 unit tests (run before any experiment)
- `run_experiments.py` main suite: 4 regimes × 7 methods × 10 seeds + ablations + capacity sweep + tariff law
- `make_analysis.py` statistics (exact sign tests), pre-registered prediction checks P1–P5, figures

## Reproduce everything
```bash
python -m pip install numpy matplotlib
python pes_tests.py                 # must print: 23 passed, 0 failed
python run_experiments.py           # ~40 min on laptop CPU; writes ../results/main_results.json
python make_analysis.py             # writes ../results/ANALYSIS.md + 6 figures
```
Quick smoke test: `python run_experiments.py --quick` (~3 min, 3 seeds).

## Design guarantees (fairness)
1. Identical linear predictor, identical SGD update, identical capacities across all capacity-limited methods.
2. Methods differ only in routing policy:
   - `pes`            full PES rule (value − tariffs − switching costs, hysteresis-gated)
   - `random_routing` same machinery, valuation replaced by noise (control C-b)
   - `clock`          same value signal, fixed-period swaps (strong fixed-profile baseline)
   - `clock_mag`      magnitude-ranked fixed swaps (classic heuristic)
   - `single_decay`   no slow store (delta-rule-like)
   - `sgd_dense/l2`   unlimited-memory references (NOT competitors — upper bound context)
3. Seeds are stable across runs/machines (`zlib.crc32(regime)` env seeding); learner seeds explicit.

## Pre-registered predictions (in research/theory/PES_FORMALIZATION.md §7, written BEFORE full run)
P1 dominance structure · P2 residence bimodality · P3 promotion↔error coupling ·
P4 reminiscence (faster re-acquisition) · P5 tariff-law monotonicity.
Competing hypotheses C-a/C-b/C-c with distinguishing tests are pre-registered in §8.

## Known limitations (honesty contract)
Linear model; tiny scale; greedy routing (near-optimal under separability, unproven otherwise);
value attribution = first-order responsibility (no credit interaction terms).
