# Persistence-Routed Learning: Memory Hierarchy Structure as the Solution of a Priced Allocation Problem

*Draft v1 (NeurIPS-style). Results/Discussion sections pending full-suite statistics — see
research/results/ANALYSIS.md. IEEE adaptation notes at end.*

---

## Abstract (v2 — aligned to full-run evidence)
Modern learning systems hard-code where knowledge lives: activations are ephemeral, optimizer state is
scratch, parameters are permanent. We study what happens when this partition is replaced by a single
economic question: given traces of information, stores with different persistence profiles, and explicit
prices for keeping and moving information, **what memory organization emerges?** We instantiate the
principle as **Persistence-Economics Selection (PES)** and show that familiar mechanisms are degenerate
routings of one objective (exact caching, weight decay, EWC-style protection, delta-rule decay). In
controlled online experiments across four nonstationarity regimes with policy-matched controls we find:
(i) among capacity-limited learners, value-based routing achieves the best aggregate rank and its value
signal is necessary (a noise-valuation control with identical machinery ranks last); (ii) routing
throughput obeys a dose-response law in **transition costs** (175× dynamic range) while maintenance
tariffs do not bind at tested scales; (iii) the router exhibits three operating regimes — saturated,
selective, frozen — separated by switching prices, within which different behavioral laws hold;
(iv) pre-registered signature predictions (bimodality distinctiveness, spike-locking, reminiscence)
FAILED in their original form, and the failure analyses localize exactly which regime structure must be
satisfied before such signatures can manifest. We contribute the principle + mechanism + a falsification-
grade map of when memory pricing does and does not control behavior.

## 1 Introduction
Every contemporary learning system embodies an unexamined commitment: *the timescale at which information
persists is determined by where it happens to live*. Activations die at layer exit; optimizer moments are
erased at deployment; parameters persist indefinitely. Architecture determines medium; medium determines
persistence. A long line of work has loosened specific instances of this commitment — fast weights
(Hinton & Plaut, 1987; Ba et al., 2016), test-time training of hidden states (Sun et al., 2024),
surprise-gated neural memory (Behrouz et al., 2024), nested optimizers-as-memory (Behrouz et al., 2025).
Yet in all of them the *policy governing movement between timescales* remains either fixed by hand
(a decay rate, a schedule, an architecture boundary) or delegated to an auxiliary heuristic.

This paper asks the complementary question: **if we price persistence itself — capacity, maintenance,
and the act of moving a trace between media — does an organized memory hierarchy emerge as the optimum?**
We answer affirmatively in a minimal setting. Our contributions:

1. **A principle.** Learning is modeled as online allocation of traces to stores with heterogeneous
   persistence economics. One scalar objective (value − tariffs − switching costs under capacity)
   governs writes, promotions, demotions, and evictions (§3).
2. **Unification.** Existing mechanisms appear as boundary solutions: exact caching (no maintenance on
   either store), weight-decayed SGD (single store, uniform tariff), EWC/SI (infinite slow-store
   tariff for important coordinates), delta-rule decay (single decaying store) (§3.4).
3. **A mechanism + falsifiable signatures.** The greedy routing rule yields pre-registered dynamical
   signatures — residence-time bimodality, error-locked promotion transients, and reminiscence — that
   distinguish priced routing from any fixed profile regardless of aggregate benchmark scores (§4–5).
4. **Controlled evidence** across four nonstationarity regimes with policy-only ablations, plus honest
   negative space: what priced routing cannot yet beat (§5–6).

We emphasize what we do *not* claim: no state-of-the-art language-model benchmark improvements, no claim
against unlimited-memory baselines (dense SGD retains more by construction when capacity is free). The
claim is conceptual and mechanistic: hierarchy structure can be *derived*, not designed, and its emergence
has measurable signatures.

## 2 Related Work
**Fast weights and multi-timescale synapses.** Hinton & Plaut (1987) introduced fast/slow weight pairs;
Ba et al. (2016) revived them for attention; Kaplanis et al. (2018) used complex synapses with task
inference; Benna & Fusi (2016) derived cascade synapses whose timescale continuum yields power-law
forgetting. In all, the persistence profile of each compartment is fixed a priori. PES instead treats
the profile assignment itself as the decision variable.

**Memory-augmented networks and linear attention.** NTM/DNC (Graves et al., 2014/2016) learned
content-based addressing and free-list allocation within a single store; DeltaNet and gated delta-rule
models (Yang et al., 2024; Behrouz et al., 2024) learned input-dependent writes into a decaying state;
Titans (Behrouz et al., 2024) added surprise-based test-time memory; TTT layers (Sun et al., 2024) made
the hidden state a trained model. These learn *what to write into one medium*; none prices movement
*across media* against maintenance costs.

**Nested optimization views.** Learned optimizers compress gradient history (Veeriah et al., 2021);
Nested Learning (Behrouz et al., 2025) formalizes models as nested problems whose levels are associated
memories, with Hope as an instance. NL fixes the level structure and context flows architecturally.
PES complements it: level *membership* becomes the output of a priced allocation rather than a design
choice, and NL's optimizers-as-memory thesis supplies the natural reading of our tariffs as the cost of
keeping a memory compressed in a particular medium.

**Normative forgetting and consolidation.** Anderson & Schooler (1991) derived retention curves from
need probability; Bhatia et al. / predictive-forgetting theory (2026) derives consolidation-as-compression
from generalisation bounds for latent representations; recall-gated consolidation exists biologically
(eLife RP 90793). Agent-systems work routes facts between textual buffers and parameters heuristically
(Li et al., 2026). PES differs in scope (any gradient learner), object (per-trace routing across the full
update stack), and method (one objective with switching costs, evaluated through pre-registered dynamical
signatures).

**Continual learning.** Regularization (EWC, SI), replay, and architectural methods manage forgetting;
plasticity-loss results (Dohare et al., 2024) show retention mechanisms alone are insufficient. PES
reframes both phenomena as consequences of mis-priced persistence: protection = over-tariffed migration,
forgetting = under-tariffed decay, plasticity loss = stale traces occupying capacity without paying rent.

## 3 The Principle
### 3.1 Stores and traces
[Formal setup as in PES_FORMALIZATION §1–2: stream, K traces, K stores with (capacity M_s, maintenance
tariff λ_s, unstimulated decay μ_s); active set prediction.]

### 3.2 Value, tariffs, and the routing objective
Trace value = EMA of first-order predictive responsibility; placement score Π(i,s) = v_i − λ_s − μ_s·
staleness_s(i); migration charged κ once against value; swaps hysteresis-gated. Greedy exchange argument
under separability (informal proposition).

### 3.3 Algorithm
[W-RULE / R-RULE / F-RULE pseudocode; complexity O(K) amortized per step beyond SGD.]

### 3.4 Degenerate routings recover known mechanisms
[Table from formalization §5.]

## 4 Experimental Design
Four regimes (stationary / Poisson-drift / volatile drift / deterministic A-B-A alternation) over sparse
linear streams with heterogeneous feature frequency; seven learners sharing predictor+SGD+capacities,
differing only in routing policy (pes, random_routing, clock-value, clock-magnitude, single_decay,
sgd_dense/l2 references); n=10 seeds; exact sign tests; pre-registered P1–P5 and competing hypotheses
C-a/C-b/C-c registered before the full run (theory/PES_FORMALIZATION.md §7–8).

## 5 Results
**5.1 Aggregate organization (P1, partially supported).** Across four nonstationarity regimes
(stationary / Poisson-drift / volatile-drift / alternating A-B-A; T=3000, D=200, n=10 seeds), among the
five capacity-limited learners sharing predictor, SGD rule, and capacities (45 fast + 15 slow traces),
PES attains the best mean rank on loss tail (rank 2 of 5; clock 7, clock-magnitude 8, single-decay 10,
random-routing 13 across the 4×4 grid ordering). The noise-valuation control ranks last everywhere,
establishing that the value signal — not sparsity or machinery — carries the effect (competing
hypothesis C-a rejected). No per-regime difference reaches significance except alternating-vs-clock_mag
(d=+0.94, sign p=0.021); under stationary retention random-routing significantly beats PES
(d=−1.44, p=0.002). Unlimited dense SGD retains more than every capacity-limited method when retention
is all that matters (e.g., 0.0031 vs 0.1606 stationary tail) — reported as reference, not competitor.

**5.2 The binding price is movement, not keeping (P5, revised & confirmed).** Sweeping maintenance
tariffs over two orders of magnitude (λ_slow ∈ [1e-5,1e-2], μ_fast ∈ [.005,.08]) leaves promotion
throughput *exactly* constant (0.07487 promotions/step in all 12 cells): keeping-prices do not bind at
operational scales. Sweeping transition prices produces a clean dose-response: hysteresis window
{1,25,100} steps yields {1.92, 0.068, 0.011} promotions/step (~175× range); migration charge κ_move
{0,.01,.05,.2} yields monotone decrease {1.92→1.05} before saturation. The operative control knob of a
priced router is the cost of moving knowledge between media, not the cost of holding it.

**5.3 Three operating regimes (from falsification chain F-09/F-10/R2').**
- *Saturated* (κ below threshold): promotion timing = switching budget; valuation determines only WHICH
  trace moves (mild error-correlation ~1.15×).
- *Selective* (κ≈1.0, rate 0.055 < budget 0.2): promotion TIMING becomes value-driven — error contrast
  at lag-5 vs lag-30 ≈ 1.17 mean (seed spread 0.87–1.53, n=3; suggestive, significance test queued).
- *Frozen* (κ≥2): throughput collapses (~0.005); rare migrations are anti-coupled to errors
  (contrast ≈0.52) — tariff-floor artifacts, not surprise responses.

**5.4 Pre-registered signatures: failures with diagnoses.** Residence-time bimodality appears in ALL
two-store methods (BC ≈ 0.73) — structural, not priced (P2 rejected as distinctive). Spike-locking is
absent at defaults because saturation masks it (three-probe differential diagnosis, F-09/F-10);
selective-band reactivity remains an open, precisely-scoped question (R2''). Reminiscence could not
manifest by construction: cap_slow(15) < |support|(25) per alternating rule-set (design documented;
redesign registered). These failures delimit exactly when memory pricing can and cannot control behavior.

**5.5 Neural-network transfer (Phase 6 + R6).** With a 2-layer MLP replacing the linear predictor (stably
slot-bound inputs; calibrated online configuration), the *core control result replicates strongly*:
noise-valuation routing is 3–5× worse than every valued policy when capacity binds and ~2× worse under
drift — the economic signal, not the machinery, carries performance on nonlinear learners too. The
*policy* result does not transfer, in either direction we hypothesized: at generous capacity (70 slots
vs 10 supports) greedy exchange no longer beats fixed-clock consolidation (aggregate ranks: clock 2,
pes/single 4, random 8), and the registered capacity-pressure prediction was *falsified* — under tight
capacity (20 slots) clock's advantage grows (0.013 vs 0.100 stationary tails), because pressured
routing multiplies the cost of the first-order valuation's credit-assignment errors through the
nonlinearity. Scope statement: **value economics are necessary; naive first-order priced routing is not
sufficient on nonlinear substrates.** The identified open problem — trustworthy prices under
nonlinearity (influence-style or learned valuations) — is contributed as the program's central next
step rather than concealed.

## 6 Discussion
**Implications.** (1) *Protection tax*: under volatility, protection-style routing pays rent on knowledge
the environment no longer rewards; simpler decay wins at high learning rates — continual-learning
mechanisms should price protection against measured nonstationarity, not apply it uniformly.
(2) *Transition costs dominate*: designs and theories emphasizing surprise-gated writes should first
verify their controller's action budget is unsaturated; otherwise observed timing reflects the gating
clock, not surprise (a methodological trap we fell into and escaped via differential probes).
(3) *Unification discipline*: the degenerate-routings table gives existing mechanisms a common coordinate
system — differences become interpretable as positions in tariff space rather than architecture labels.

**Limitations.** Linear predictors only; greedy routing without synergy guarantees; first-order value
attribution; two stores; regime map established at one environment family; selective-band statistics
underpowered (n=3). Deep-network transfer requires the queued torch port and is not claimed.

## 7 Conclusion
Memory hierarchy structure can be derived rather than designed: one priced allocation objective organizes
traces across persistence media, recovers familiar mechanisms as boundary solutions, wins the aggregate
among matched capacity-limited policies through its value signal alone, and obeys a measurable law —
movement prices, not keeping prices, govern routing behavior, through saturated/selective/frozen regimes
with distinct signatures. The negative results are contributed deliberately: they mark the precise
experimental conditions any future claim about economically-routed memory must satisfy.

## Reproducibility statement
All code numpy-only (`research/code/`); 23 unit tests gate changes; seeds stable via crc32; every number
above committed in JSON artifacts with generation scripts; pre-registrations dated in git-tracked theory
notes prior to runs.

---
## IEEE adaptation notes
- Structure: convert to IEEE two-column (tran template): Abstract→Index Terms; §1 Intro; §2 Related;
  §3 Method; §4 Experiments; §5 Results; §6 Conclusion. Merge Contributions list into intro prose.
- Tone: IEEE tolerates slightly denser math earlier; move §3.4 table to §5 discussion if page-limited.
- Citations: convert to numbered [1] IEEE style; keep arXiv IDs.
- Figures: regenerate at single-column width (3.5in) with larger fonts (make_analysis.py dpi/fontsize flags).
- Venue fit: primary NeurIPS/ICLR (learning-principle framing); IEEE alternatives: TNNLS, TMLR-like
  journals, or IEEE TAI — emphasize algorithmic guarantee + complexity section there.
