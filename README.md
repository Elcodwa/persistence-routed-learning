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
kv_pes/                 EMPIRICAL TEST: the paper's PES rule applied to LLM KV-cache
                        eviction (Theorem 1 margin rule vs H2O / StreamingLLM /
                        sliding-window / random, needle-in-haystack + multi-fact QA,
                        GPT-2 / TinyLlama / Qwen2.5 on Colab T4). Entry point:
                        kv_pes/notebooks/colab_runner.ipynb; results template and
                        pre-registered expectations in kv_pes/results/README.md.
pes_core.py            core environment + learners (numpy only)
pes_tests.py           23 unit tests gating every change
run_experiments.py     main suite: regimes x methods x seeds + ablations + sweeps
make_analysis.py       statistics + pre-registered prediction checks
make_paper_figures.py  publication figures (PDF)
pes_torch.py           MLP port (torch); r6_capacity.py capacity-pressure sweep
followup_probes.py     registered follow-up probes (R2/R4)
r1_p4_redesign.py      P4 reminiscence redesign (cap_slow >= 2*|support| + guard)
r9_scaleup.py          R9 true-scarcity scale-up (D=800, s0=50, caps 110+55)
r7_mlp.py              R7-MLP: exact LOO deletion re-score at routing moments (MLP port)
../../paper/latex/     LaTeX source of the paper + figures
                       (tmlr_main.tex = TMLR format; main_twocolumn.tex =
                        extended 13-page two-column journal version with
                        full proofs, architecture diagrams, and applications)
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
# publishability follow-ups:
python r1_p4_redesign.py    # P4 redesign probe (~2 min)
python r9_scaleup.py        # true-scarcity scale-up (~7 min)
```

## Status & honest scope

**Headline result:** routing performance tracks price quality, not routing structure — swapping cheap
gradient tags for exact leave-one-out loss prices halves retention error (p=0.002), ties
fixed-schedule consolidation, and keeps noise controls strictly worst, with all machinery fixed.
Laws are measured on sparse linear streams (two feature families) plus a two-layer MLP port.
Dense SGD + weight decay remains the reference when memory is not truly scarce; priced routing targets
genuinely capacity-bound regimes (KV budgets, edge streams, modular routers). Negative results are
reported with diagnoses in the paper. See `research/results/R7_R8_REPORT.md` and
`research/results/RESULTS_NARRATIVE.md`; full theory in `research/theory/`.

## License

MIT — see [LICENSE](LICENSE).
