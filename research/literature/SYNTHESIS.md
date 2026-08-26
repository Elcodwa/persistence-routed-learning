# SYNTHESIS — What the Field Solves, Assumes, Fails, and Leaves Unexplained

Answering, per major direction: (1) problem solved, (2) abstraction, (3) assumptions,
(4) fundamental limitation, (5) unexplained residue, (6) contradictions, (7) surprisingly weak results,
(8) abandoned ideas worth reviving, (9) promising-but-incomplete, (10) unanswered questions.

## 1. Attention-era sequence modeling (Transformer/SSM/linear-attention/fast-weights)
1. Solves: parallelizable context mixing; linear-cost approximations thereof.
2. Abstraction: KV pairs as key-addressed values; recurrent forms as state + rank-constrained writes.
3. Assumptions: the STORE (state matrix, cache, hidden state) has an architecturally FIXED persistence
   profile (exact copy forever, or fixed/input-gated decay); the UPDATE RULE CLASS (softmax attn, delta
   rule, convolution, polynomial basis) is chosen a priori.
4. Limitation: no principled way to decide what deserves persistence; interference grows with load;
   effective-context cliffs; recency bias.
5. Unexplained: why phase changes (induction heads) emerge when they do; why exact-copy caches still show
   position bias; what determines the empirical "effective memory horizon".
6. Contradictions: Hopfield-vs-attention framings disagree on whether memory is retrieval or computation;
   "ICL=GD" vs its rebuttals.
7. Weak: linear-attention recall degrades sharply on multi-step associative recall despite theory-friendly
   forms (recency/over-smoothing analyses).
8. Abandoned: fast weights (1991, 2016) were premature — hardware, not concept, failed them.
9. Promising-incomplete: Titans/TTT/GatedDeltaNet — test-time gradient memory, but persistence/gating
   policy classes remain hand-designed; no normative account of WHAT should migrate where.
10. Unanswered: is there ONE quantity that all these stores (KV, state, moments, weights) optimize?

## 2. Continual learning / plasticity
1–3: solves sequential-task retention via regularization/replay/architecture; assumes forgetting is THE
   enemy; stability-plasticity treated as tradeoff to balance.
4: methods either protect (SI/EWC) or erase-and-replace; almost none MOVE knowledge between timescales.
5: plasticity loss persists even without forgetting (primacy bias); why protection itself destroys future
   learning is unclear.
6: replay helps retention but hurts plasticity in some regimes; EWC fails silently under drift.
7: many CL benchmarks solvable by shortcut (task-id cues); weak baselines common.
8: Hinton&Plaut fast/slow weights (1987) — revived only in fast-weight line, not as CONSOLIDATION.
9: complex synapses (Kaplanis) hint at multi-timescale variables but task-inference machinery dominates.
10: what is the correct OBJECTIVE for a lifelong learner? (cumulative regret? compression? both?)

## 3. Optimization
1–3: solves noisy high-dim descent; assumes optimizer state is scratch (moments = statistics of gradients,
   not knowledge); assumes a SINGLE global timescale per moment tensor (β1, β2).
4: state discarded at deployment = discarding compressed experience; loss spikes at scale remain partly
   unexplained (Adam instability theory attributes to time-correlated curvature, mitigation empirical).
5: why do learned optimizers fail to generalize across scales/tasks? (meta-overfitting unresolved.)
6: SAM helps though flatness correlates poorly sometimes; Muon spectral story vs classical momentum views.
7: learned optimizers rarely beat tuned Adam outside their meta-training distribution.
8: Kepler–Hinton "optimizers as associative memory" intuition existed informally; Nested Learning made it
   formal but did not derive WHICH memories deserve which persistence.
9: Nested Learning (Hope): expressive optimizers as deep memory — but hierarchy structure hand-set.
10: is there an optimal ALLOCATION of information across (context, optimizer state, weights)?

## 4. Train/inference boundary
1–3: solves static deployment; assumes inference is read-only w.r.t. long-term state (except TTT-style
   within-sequence state).
4: test-time training/adaptation methods adapt state but discard it after the sequence; no cross-session
   accumulation norm; no accounting of the VALUE of an inference episode for future performance.
5: ICL works without weight change — but WHY the same capability can't be retained afterwards without
   expensive fine-tuning lacks a quantitative theory (the "retention gap").
6: ICL≈implicit-GD claims vs rebuttals (approximation only in restricted settings).
7: test-time scaling gains saturate; sampling tricks don't compound into durable skill.
8: prequential/online evaluation philosophy (statistics community) largely ignored by ML practice.
9: TTT layers, Titans — inference-time writes exist but are per-sequence ephemera.
10: can inference episodes be priced in the SAME currency as weight updates?

## 5. Consolidation (neuro-inspired ML)
1–3: CLS explains hippocampus/cortex split; generative replay mitigates forgetting; assumes consolidation
   is scheduled replay + gradual weight change.
4: transfer POLICY (when/what/how-much to move between stores) is hand-designed or biologically asserted.
5: predictive-forgetting theory (2603.04688) explains WHY compress during consolidation (generalisation),
   but treats representation latents, not the parameter/update stack; no online decision rule.
6: sleep beneficial in some models, harmful (over-consolidation) in others — no unified criterion.
7: most ML "consolidation" = EMA of weights (mean teacher) — a fixed-rate heuristic with no normative base.
8: cascade models (Fusi; Benna–Fusi) — beautiful multi-timescale synapse theory, barely transplanted to DL.
9: GENESIS-type episodic-semantic interaction models — conceptual, not yet a training principle.
10: is there a single control law governing information flow across ALL persistence timescales?

## CROSS-CUTTING UNIFICATION ATTEMPTS FOUND (nearest existing syntheses)
- Linear attention ≡ fast weights ≡ Hopfield update (three views, one algebra) — unifies READ/WRITE
  algebra, not persistence policy.
- Nested Learning: unifies optimizer/model/memory as nested learners — but levels & flows are designed.
- Predictive forgetting: normative compression for consolidation — but single-store (latents), offline.
- HiPPO/S4: fixed polynomial timescale bases — elegant but non-learned placement.
- Two-timescale SA: convergence theory for fixed hierarchies — no learned allocation.
- Adaptive Kalman / volatility-adaptive gains: optimal single-store tracking under unknown drift — no
  multi-store economics, no connection to NN memory stacks.

## THE GAP THIS EXPOSES (working hypothesis for the project)
Across all five directions, every method fixes (by architecture or hyperparameter) the PERSISTENCE
PROFILE of its stores, then learns contents. Nobody appears to learn, from one objective, the ROUTING of
information across persistence timescales spanning activations → optimizer state → weights, with an
explicit maintenance cost. If such a principle existed, several existing mechanisms should fall out as
degenerate cases (fixed routing), and new capabilities (boundary-free continual learning, reminiscence
transients, bimodal persistence spectra, spike/migration coupling) should become predictable.
Novelty status: APPARENTLY NOVEL at synthesis level (components individually well-covered);
adversarial verification pending.
