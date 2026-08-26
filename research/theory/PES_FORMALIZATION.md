# PES — Persistence-Economics Selection: Mathematical Formalization (v1)

## 1. Setting
Stream (x_t, y_t), t=1..T. Learner maintains K traces w = {w_i}. Prediction f(x_t; A_t) uses only the
ACTIVE set A_t ⊆ w selected by current store assignment. Stores S = {fast, slow} (extendable) with:
- capacity M_s (max traces resident),
- maintenance tariff λ_s per step per resident trace (λ_fast ≪ λ_slow),
- survival time-constant τ_s under no-use decay (τ_fast ≪ τ_slow).

## 2. The Objective (one scalar functional)
Minimize prequential loss plus storage/maintenance/transition costs:
  J = Σ_t ℓ(y_t, f(x_t; A_t))  +  Σ_s λ_s · Σ_{i∈s} dt  +  κ·(#transitions)  s.t.  |s| ≤ M_s ∀s,
with ℓ = ½(y−ŷ)². No term refers to an architecture; stores differ ONLY in (λ_s, τ_s, M_s).

## 3. Per-trace valuation (the sufficient statistic)
Each trace carries running value = discounted magnitude of prediction-error it is accountable for:
  v_i(t) = ρ·v_i(t−1) + (1−ρ)·|x_{t,i}·(y_t − ŷ_t)|        [linear-model instantiation]
General nets: v_i = EMA of |∇_{w_i} ℓ·w_i| (first-order responsibility). Cost grows with norm:
  c_i = c₀ + α‖w_i‖₁. Utility-per-cost: u_i = v_i / (c_i + ε).

## 4. Routing rule (greedy is optimal here)
Define persistence score of trace i if placed in store s:
  Π(i,s) = u_i − λ_s − μ_s·age_s(i),      age-sensitivity μ_s = 1/τ_s.
Routing policy each step:
  W-RULE (write): new trace enters cheapest store with free capacity; else replaces argmin_{fast} Π.
  R-RULE (promotion/demotion): let i⁺ = argmax_{i∈fast} Π(i,slow), j⁻ = argmin_{j∈slow} Π(j,slow);
     swap iff Π(i⁺,slow) > Π(j⁻,fast)   (and symmetrically for demotion).
  F-RULE (forget): delete i iff max_s Π(i,s) < 0.
Proposition (informal): with additive costs and separable utilities, greedy exchange (R-RULE) is optimal
per-step; standard argument via assignment-problem matroid exchange property. Full proof deferred.

## 5. Existing mechanisms as degenerate routings (unification table)
| Mechanism            | Recovered by                                                        |
|----------------------|---------------------------------------------------------------------|
| Exact cache / KV     | M_fast→∞, λ_slow→∞, τ_slow→∞ (never promote/forget)                 |
| AdamW                | single store, λ=weight-decay rate; moments = auxiliary value stats   |
| EWC/SI protection    | λ_slow→∞ for protected coords ⇒ Π<0 for any perturbing transition    |
| Delta-rule decay     | single store, μ = gate; writes ungated                              |
| Fixed consolidation  | R-RULE replaced by clock-driven swaps (our baseline "fixed-two")     |
| No memory (SGD)      | M_fast = D, λ=0, no routing bookkeeping                             |
Thus designed hierarchies = boundary points of tariff space; PES interpolates/prices the interior.

## 6. Complexity
Per step: value update O(K), routing O(K) (two heaps), prediction O(|A_t|). Total O(T·K), memory O(K+D_ref)
— vs O(D) state for dense Adam; active set often |A_t| ≪ D under load.

## 7. Falsifiable predictions — VERDICTS AFTER FULL RUN (10 seeds)
- **P1 dominance structure: PARTIALLY SUPPORTED.** Aggregate ranks among capacity-limited family:
  PES best (rank 2), clock 7, clock_mag 8, single_decay 10, random_routing 13. No fixed profile beats
  PES on ≥3/4 regimes (kill criterion NOT triggered), but no per-regime win reaches significance
  except alternating vs clock_mag (d=+0.94, p=0.021). Dense SGD remains upper reference everywhere.
- **P2 bimodality: REJECTED AS DISTINCTIVE.** Residence distributions ARE bimodal (PES BC=0.731,
  clock BC=0.729 > 0.555) but BOTH methods show it ⇒ property of two-store bookkeeping, not of pricing.
- **P3 spike-locking: NOT SUPPORTED at default settings** (lag-5 error ratio 1.01 vs clock 1.12).
  Theory now explains why: hysteresis throttles reactivity; coupling should re-emerge at small windows.
- **P4 reminiscence: FAILED AS DESIGNED.** Reacquire ≈ acquire (1.4 vs 2.0 steps, ns). Root cause
  identified: cap_slow=15 cannot hold both A/B rule supports (25 coords each) ⇒ nothing survives visits.
  Redesign required (larger slow store / longer blocks / sparser rules) before the claim can be tested.
- **P5 tariff law: REVISED & CONFIRMED IN REVISED FORM.** Maintenance tariffs (λ_slow up to 1e-2,
  μ_fast up to .08) leave promotion rate EXACTLY flat (0.07487 across all 12 cells) ⇒ they do not bind.
  Transition costs bind strongly: hysteresis {1,25,100} → 1.92/0.068/0.011 promo/step; κ_move
  {0,.01,.05,.2} → 1.92/1.80/1.72/1.05 (hw=1); saturation where hysteresis floor binds.
  ⇒ The operative price of persistence in this mechanism is the COST OF MOVING, not of keeping.

## 8. Competing hypotheses verdicts
C-a capacity-only explanation: REJECTED (random_routing shares capacities/sparsity yet ranks last;
value signal matters). C-b any-gating-suffices: PARTIALLY SUPPORTED (clock close to PES in several
regimes) — valuation quality contributes but does not dominate loss-tail metrics. C-c optimization-luck:
not fully excluded pending LR sweeps; flagged as revision item.

## 8b. Pre-registered competing hypotheses (original registration, kept for the record)
- **C-a (capacity):** PES gains come merely from sparsity (fewer effective params) under load.
  Test: sparse-matched SGD-L1 baseline at identical active-set trajectory.
- **C-b (any-gating):** ANY learned scalar gating beats fixed rates; PES specifics irrelevant.
  Test: ablate valuation (random u_i), keep all machinery.
- **C-c (optimization luck):** differences vanish under tuned per-method LRs.
  Test: LR sweep per method, compare best-vs-best.
Distinguishing outcomes recorded in results/COMPETING_HYPOTHESES.md.

## 9. Threats to validity
Tiny scale; linear model may not transfer to deep nets; value-attribution error under correlated features
(release: correlation-controlled envs); greedy ≠ global optimum under synergy (future work: submodular
relaxation). Novelty language remains "apparently novel" pending deeper sweeps.
