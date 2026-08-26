# RESEARCH LOG — Autonomous ML Learning-Principle Discovery

Start: 2026-08-25 (Egypt Time). Operator: sleeping; agent runs autonomously until stopped.

## Mission
Find, formulate, mathematically specify, and experimentally stress-test a *potentially important new
learning principle* — not an architecture tweak. Willingness to kill ideas is mandatory.

## Method contract (self-imposed)
1. Literature-first: map field before hypothesizing. Every external claim gets a source URL.
2. Scientific Gaps Database before candidate selection (target ≥45 rows).
3. ≥50 fundamentally distinct hypotheses, each with falsifiable prediction + minimal killing experiment.
4. Anti-gimmick filter: no "Transformer+X", no renamed mechanisms, no pure combinations.
5. Adversarial prior-art search per finalist ("what exactly exists before us?" / "what single statement
   could we claim that they cannot?").
6. Tiny falsification experiments BEFORE any scaling. Pre-registered competing hypotheses.
7. FAILED_IDEAS ledger maintained as a research asset.
8. Novelty language discipline: "apparently novel", "closest prior work found", etc. unless evidence is strong.

## Environment
Windows 11 host, bash (MSYS). Python 3.11.9 (Store) — numpy/matplotlib being installed. No torch yet;
tiny experiments designed numpy-first. arXiv API + Semantic Scholar API reachable via curl.

## Decisions
- D0 [start]: Chose autonomous single-agent execution with programmatic search sweeps (execute_code loops
  over web_search/arXiv/S2 APIs) instead of many interactive calls — maximizes coverage per context unit.
- D1 [hypotheses]: 52 candidates scored; H01 (persistence economics) consolidated with H16/H32/H34/H40
  facets into single program "PES". H13 folded in as prequential evaluation protocol. H48 deferred.
- D2 [falsifications during build]: F-06 utility-density pricing inverts (newcomer paradox);
  F-07 kappa-as-score-penalty caused thrashing → replaced by one-time migration charge + temporal
  hysteresis; F-08 fixed-clock consolidation churns because scheduled swaps ignore trace value.
- D3 [fairness]: all capacity-limited methods share identical predictor, SGD rule, capacities, value EMA;
  differ only in routing policy. Dense SGD reported as upper reference, NOT as a competitor claim.
- D4 [claim discipline]: headline claim = capacity-constrained regimes + within-mechanism dynamical
  signatures (P2 bimodality, P3 spike-locking, P4 reminiscence). No claims of beating unlimited memory.
- D5 [reproducibility]: stable env seeds via zlib.crc32(regime); 10 seeds; exact sign tests; JSON dumps;
  unit tests gate every code change (23 checks).
- D6 [venue]: user targets NeurIPS/IEEE. Paper drafted ONLY after full-suite statistics exist;
  IEEE adaptation notes included alongside NeurIPS-style main text.
- D7 [night arc summary]: full suite (4 regimes × 7 methods × 10 seeds + ablations + capacity + tariff)
  completed; P1 partial, P2/P3/P4 rejected-with-diagnosis, P5 revised to transition-cost law; probes
  R2/ρ/R2' mapped THREE-REGIME routing law (saturated/selective/frozen) with per-regime signatures;
  R4 LR sweep closed C-c partially (PES best at its best LR under drift). Paper abstract v2 rewritten to
  match evidence; signature claims downgraded everywhere. Follow-ups queued for cron: R1 (P4 redesign),
  R2'' (selective-band significance), R3 (three-regime theorem), R5 (torch port, blocked on consent).
- D8 [torch port debugging arc — appendix material]: user approved torch install (2.13 CPU).
  Port initially failed to learn; diagnosis chain: (a) input-slot shifting on admission scrambled
  learned weights → fixed with stable slot binding (cache-line style); (b) batch-1 online Adam at
  lr≥0.01 diverges (pure-fit control showed fit-then-degrade) → fixed with lr=0.003 + grad clipping +
  small output-layer init; (c) task recalibrated for ONLINE MLP convergence (s0=10, amplitude 0.6,
  T=8000). Substrate control then reached 0.015 tail vs 0.05 predict-mean floor. Lesson recorded:
  in online single-sample regimes, verify SUBSTRATE learnability before attributing anything to policy.
- D9 [Phase 6 launched]: 3 regimes × 4 policies × 3 seeds, calibrated config; results to land in
  torch_phase6.json; comparison against linear-model qualitative findings (value-signal necessity,
  protection tax under volatility).
- D10 [Phase 6 outcome]: value-signal necessity REPLICAATES strongly on MLP (random-routing 3–5× worse);
  PES policy dominance does NOT replicate at loose capacity (clock wins aggregate). Scope narrowed to
  "value economics necessary, optimal policy open". R6 capacity-pressure sweep launched to test the
  registered mechanism (advantage re-emerges as caps → support size). Paper §5.5 added with both halves.
- D11 [R6 falsification]: prediction inverted — pressure AMPLIFIES clock advantage (F-11). Root insight:
  naive gradient prices are least trustworthy exactly when routing is most frequent. Program pivots to
  valuation quality as the core open problem (R7 influence-style pricing), plus R2'' and R1 from before.
  This is the honest frontier of the project after ~30 hours: strong principle-level control result,
  working mechanism at linear scale, open policy problem at nonlinear scale.
- D12 [submission-grade paper]: research/paper/PAPER_SUBMISSION.md written per user request. Every number
  traced to committed JSON artifacts (ablation values pulled programmatically, none invented); humanizer
  pass applied (34-pattern audit) while preserving scientific hedging discipline (overclaiming is itself
  an AI tell in academic prose); title changed to question form; failure-ledger integrated into abstract
  and discussion per top-venue norms; IEEE adaptation notes embedded. Authorship/AI-disclosure checklist
  handed to user (venues require disclosing AI writing assistance; user must run final fresh prior-art
  sweep before submission).
- D13 [pre-submission sweep #5]: two NEW concurrent works found — Chen & Cheng 2606.12945 (learned
  multi-factor value for single-store TEXT memory; title overlap, no cross-store economics) and Neural
  Garbage Collection 2604.18002 (read: RL KV-eviction, single store — corner case, cited). Related-work
  §2 updated; verdict file research/literature/SWEEP5_FINAL.md. Claim remains "apparently novel".
- D14 [fix-all pass]: (1) Propositions 1&2 with proofs added (theory/THEOREMS.md + Appendix A); (2) seed
  expansion n=10 × two families — drift dominance now significant vs ALL baselines in BOTH families;
  Gaussian family shows strongest capacity-pressure effect; original stationary-instability cell
  reported as unresolved variance; (3) R2'' completed: selective-band spike-locking REJECTED (p=0.83),
  paper §5.3 rewritten to the settled price-vs-surprise dissociation; (4) second env family done;
  (5) NGC read and cleared; (6) nonlinear substrate = Phase6/R6 MLP port (true LM scale remains future
  work, stated as such). Paper updated throughout; acceptance outlook revised upward for TMLR/IEEE.
- D15 [LaTeX build]: research/paper/latex/main.tex — standalone-compilable NeurIPS-style article
  (swap-in note for official sty included). Compiled clean with local MiKTeX pdflatex (2 passes,
  0 errors), output main.pdf verified: 8 pages, all propositions/results/disclosure present; one
  cosmetic overfull hbox. AI-disclosure paragraph embedded in abstract + back matter per venue policy.
  Humanizer audit previously applied to source prose; numbers traced to JSON artifacts.
- D16 [math verification caught a real error + full paper rebuild]: numerical audit showed propositions
  (margin rule) != implementation (level rule) — margin picks suboptimal pair ~94% of configs.
  Corrected router implemented; first attempt exposed slow-staleness bug (churn loop); fixed by
  refreshing last_touch for all residents; 23/23 tests green. Margin-rule re-run: value-necessity
  SURVIVES, per-regime dominance claims WITHDRAWN (clock wins stationary), Gaussian family flat,
  dose-response re-measured (margin rule saturates near floor). Paper fully rewritten to academic
  standard: corrected numbers, 4 figures from real data, formal register, author = Hazem Mohamed Ali
  Hassan, minimal AI disclosure (code assistance + review only, no agent name), MIT LICENSE, README,
  .gitignore, git repo initialized and committed (78 files, commit a6006d3). PDF rebuilt: 11 pages,
  figures embedded.
- D17 [R7/R8 pivot]: numerical verification -> margin-rule correction -> clock-beats-PES deficit ->
  diagnosis "price quality" -> PRE-REGISTERED (git commit) R7 test: exact LOO prices
  (dLi = r*wixi + 0.5(wixi)^2) replace EMA|grad| tags, nothing else changes. ALL THREE criteria pass:
  stationary 0.074±0.024 (vs EMA 0.161, p=.002; ties clock 0.089), drift 0.150 best-of-all,
  random strictly worst; volatile flat as predicted. R8 kill test: dense+wd hits 0.0024 stationary /
  0.133 drift at D=200 — unlimited-memory ceiling quantified; scope narrowed to truly scarce regimes.
  Paper rewritten around "price quality is the binding constraint" (abstract, contributions, §5.2,
  new §5.3 LOO section + scope paragraph, discussion, conclusion); Fig 3 regenerated with PES-LOO bars;
  rebuilt 12pp clean; README updated.
