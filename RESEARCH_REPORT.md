# RESEARCH DISCOVERY REPORT
## Project: Persistence-Routed Learning (PES) — Autonomous Research Cycle, Aug 25 2026

Status legend: ✅ complete · 🔄 pending final experimental statistics · ⚠️ honest limitation

---

## 1. Major unsolved problems discovered ✅
Consolidated from a 48-row Scientific Gaps Database (`research/gaps/SCIENTIFIC_GAPS_DB.md`);
the deepest cross-cutting gap:
> **No learning system derives its own memory-hierarchy structure.** Every existing method fixes,
> by architecture or hyperparameter, where information lives and how long it stays; then learns contents.
> The routing of knowledge across persistence timescales (activations ↔ optimizer state ↔ parameters)
> has no normative account, despite ≥15 independent subfields converging on pieces of one.

Secondary unsolved problems logged: optimizer-state discarding at deployment (experience loss);
plasticity loss without forgetting; why fixed consolidation schedules churn; unexplained position/recency
pathologies in exact caches; retention gap between in-context skill and parametric knowledge.

## 2. Research landscape ✅
Mapped in `research/literature/FIELD_MAP.md` (≈95 programmatic queries + targeted full-text reads) and
`SYNTHESIS.md` (per-direction: solved/assumed/fails/unexplained/contradicted/abandoned).
Key unification attempts found & positioned against: Nested Learning (fixed nesting),
Predictive Forgetting (latent-only compression norm), HiPPO/S4 (fixed bases), two-timescale SA
(fixed separation), fast-weight/Hopfield algebra (read/write math, not policy), agent memory lifecycle
systems (heuristic pipelines).

## 3. Hypotheses generated ✅
52 fundamentally distinct candidates with principle/math/prior/difference/falsifiable-prediction/minimal-
experiment each: `research/hypotheses/HYPOTHESES_LEDGER.md`. Families: persistence abstraction,
optimization reimagined, train/inference unification, self-modification, dynamics, information theory,
Bayesian, neurocomputational, economic/game-theoretic, conceptual reframes.

## 4. Rejected hypotheses ✅
`research/rejected_ideas/FAILED_IDEAS.md` — F-01 single-store write gating (prior art saturates it);
F-02 Nested-Learning overlap (constraint, not blocker); F-03 IB-consolidation rederivation;
F-04 superposition stores (renamed Hopfield mixture); F-05 deferred trio (NAS-topology, Landauer pricing,
categorical learners); F-06 utility-density valuation (**empirically falsified**: newcomer paradox);
F-07 κ-as-score-penalty (thrashing); F-08 fixed-clock consolidation churn (validated as baseline failure).

## 5. Top 10 candidates ✅
H01 8.35 · H16 7.55 · H13 7.30 · H32 7.00 · H34/H40 6.75 · H48 6.50 · H22/H25 6.35 · H09/H41 6.30
(scoring rubric & weights in ledger header).

## 6. Top 3 and selection ✅
C1 PES (selected) · C2 prequential training objective (absorbed as evaluation protocol) ·
C3 partial-evaluation view (deferred, no empirical handle). Rationale: C1 subsumes four top-ten facets;
strongest unification power + falsifiability + feasibility.

## 7. Prior-art analysis ✅ (as far as search could reach)
`research/literature/PRIOR_ART_GRAPH.md`: nearest 24+ works graphed by shared mechanism/objective/
formalism; four adversarial synonym-sweeps including post-selection wave 4. Nearest neighbors and the
exact residual claim documented ("what exists before us" / "single statement none of them can make").
Novelty language: **apparently novel** as principle+mechanism+signatures; components individually
well-covered; risk of unread identical derivation acknowledged.

## 8–9. Final hypothesis: mathematical formulation & algorithm ✅
`research/theory/PES_FORMALIZATION.md`: objective J (prequential loss + maintenance tariffs + transition
costs under capacity), per-trace valuation, greedy exchange rule with switching charge + hysteresis,
complexity O(K)/step, degenerate-routings table recovering exact caching / weight decay / EWC /
delta-rule / scheduled consolidation. Algorithm implemented in `pes_core.py` (23 unit tests green).

## 10–11. Experimental plan & predicted outcomes ✅ → FINAL VERDICTS ✅
Pre-registered BEFORE full run; verdicts after 10 seeds + follow-up probes
(full detail: results/RESULTS_NARRATIVE.md):
- P1 dominance structure: PARTIALLY SUPPORTED (PES best aggregate rank among capacity-limited family;
  value signal necessary via random-routing control; no per-regime significance except one).
- P2 bimodality distinctiveness: REJECTED (both stores show it ⇒ structural, not priced).
- P3 spike-locking: NOT SUPPORTED as registered; three-probe investigation (R2, ρ, R2') mapped the
  saturated/selective/frozen regime structure; selective-band reactivity suggestive (contrast ≈1.17)
  but not significant at n=3 — R2'' queued.
- P4 reminiscence: FAILED AS DESIGNED (slow store too small to span A/B blocks); redesign queued.
- P5 tariff law: REVISED & CONFIRMED — transition costs bind (175× dose-response), maintenance tariffs
  do not; frozen band shows anti-coupled timing.

## 12. Falsification tests ✅
Five falsifications executed and recorded (F-06..F-10), including two of my own retrodictions —
the project's most valuable epistemic moments. All baselines policy-matched; controls killed the
capacity-only explanation.

## 13. Novelty argument ✅ (as far as four adversarial sweeps reach)
Apparently novel as principle+mechanism+regime-law; nearest works documented with residual-claim graph.
Before any submission: fresh sweep required (queued in cron prompt).

## 14. Potential limitations ⚠️
Linear-model scale only; greedy ≠ global optimum without synergy bounds; first-order responsibility
attribution; two stores (extensible); signatures demonstrated in-vitro — transfer to deep architectures
is future work requiring torch-scale infrastructure.

## 15. Potential applications
Optimizer states that persist across deployments (moment transfer under explicit pricing);
continual-learning systems whose protection budget is spent by value, not schedule; cache/KV management
with learned eviction tariffs; LLM-agent memory lifecycle grounded in one objective instead of heuristics;
diagnostic signatures (bimodality, spike-locking) measurable in existing trained models' bookkeeping.

---
## Deliverable map
| Mission item | Artifact |
|---|---|
| A. Report | this file |
| B. Prototype | research/code/{pes_core,pes_tests,run_experiments,make_analysis}.py + README |
| C. Results | research/results/main_results.json, ANALYSIS.md, full_run_log.txt |
| Figures | research/figures/fig1..fig6 |
| D. Paper | research/paper/DRAFT_PES_v1.md (+IEEE notes) |
| Process | RESEARCH_LOG.md (decision log D0–D6) |
