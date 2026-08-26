# Persistence-Routed Learning (PES)

**What should a learning machine remember?** This repository contains the research code, experiments,
and paper for *Persistence-Routed Learning: Memory Hierarchy Structure as the Solution of a Priced
Allocation Problem*.

A learner maintains many traces of information and a set of stores with different persistence profiles
(decay rate, maintenance tariff, capacity). Each trace is routed to the store where its accumulated
responsibility for prediction error best covers its costs; moving between stores costs a fee. The paper
shows that familiar memory mechanisms (exact caching, weight decay, EWC-style protection, delta-rule
decay) appear as boundary points of one priced objective, proves greedy exchange optimal under stated
conditions, and measures what actually governs routed memory: movement prices dominate keeping prices,
and naive price discovery through nonlinear predictors is the binding open problem.

## Repository layout

```
pes_core.py            core environment + learners (numpy only)
pes_tests.py           23 unit tests gating every change
run_experiments.py     main suite: regimes x methods x seeds + ablations + sweeps
make_analysis.py       statistics + pre-registered prediction checks
make_paper_figures.py  publication figures (PDF)
pes_torch.py           MLP port (torch); r6_capacity.py capacity-pressure sweep
followup_probes.py     registered follow-up probes (R2/R4)
../../paper/latex/     LaTeX source of the paper + figures
../results/            committed JSON artifacts for every number in the paper
```

## Reproduce

```bash
pip install numpy matplotlib          # torch only for the MLP port
python pes_tests.py                   # must print: 23 passed, 0 failed
python run_experiments.py             # ~40 min laptop CPU -> ../results/main_results.json
python make_analysis.py               # statistics + prediction checks
python make_paper_figures.py          # figures -> paper/latex/figs/
# optional neural port:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python pes_torch.py && python r6_capacity.py
```

## Status & honest scope

All laws are measured on sparse linear streams (two feature families) plus a two-layer MLP port.
Priced routing provably allocates optimally given prices; making prices trustworthy through
nonlinearity is open. Negative results are reported with diagnoses in the paper. See
`RESEARCH_REPORT.md` and `research/results/RESULTS_NARRATIVE.md` for the full story, including the
failed-prediction ledger.

## License

MIT — see [LICENSE](LICENSE).
