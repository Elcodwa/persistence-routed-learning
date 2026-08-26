# What Should a Learning Machine Remember?

**Persistence-Routed Learning: Memory Hierarchy Structure as the Solution of a Priced Allocation Problem**

*Anonymous submission draft. Code and full result artifacts in the accompanying repository.*

---

## Abstract

A trained network has no say in where its knowledge lives. Activations vanish at the end of a forward
pass, optimizer moments are treated as disposable scratch, and weights persist forever; which information
goes where is settled by whoever wrote the architecture, not by anything the learner figured out. We ask
what happens when that settlement is withdrawn: give a learner many traces of information, stores with
different persistence profiles (decay rate, maintenance cost, capacity), and prices for keeping and for
moving information, then watch what organization emerges. The resulting rule, Persistence-Economics
Selection (PES), assigns each trace to the store where its accumulated responsibility for prediction
error best covers its costs, with an explicit fee for moving between stores. Four findings from
controlled online experiments: (1) known memory mechanisms drop out as boundary cases of one objective;
(2) across four nonstationarity regimes, value-based routing holds the best aggregate rank among
capacity-matched policies, and a noise-valuation control proves the signal itself, not the machinery,
does the work; (3) routing throughput obeys a dose-response law in transition costs spanning 175×, while
maintenance tariffs never bind at tested scales, so the operative price of memory is movement, not
storage; (4) three pre-registered signature predictions failed, and the failure analyses locate exactly
which regime conditions must hold before such signatures can appear. A neural-network port replicates
the control result but falsifies our own follow-up about capacity pressure: naive gradient-derived
prices get worse under pressure because credit assignment through a nonlinearity is least reliable
exactly when routing is most frequent. Trustworthy valuation under nonlinearity is the open problem this
program hands to itself.

## 1 Introduction

Every deployed learning system embodies a decision it never made. Information sits in activations that
die at layer exit, in optimizer state that is erased at deployment, or in parameters that outlive
everything else. The timescale at which a fact persists follows from where it happens to sit, and where
it sits was fixed before training began.

A good deal of work loosens specific instances of this. Fast weights (Hinton & Plaut, 1987; Ba et al.,
2016) gave networks a second, quicker set of parameters. Test-time training (Sun et al., 2024) turns the
hidden state into a model that learns during inference. Titans (Behrouz et al., 2024) writes to a neural
memory gated by surprise, and Nested Learning (Behrouz et al., 2025) reads optimizers as associative
memories arranged in nested problems. In all of these, something important stays fixed by hand: the decay
rate, the schedule, the level structure. The learner does not decide how long things last. It inherits
that.

This paper withdraws the inheritance. We hand the learner traces of information and a small menu of
stores, each with a persistence profile expressed as prices: what a trace costs to keep per step, how
fast it decays unattended, how much room the store has. Moving a trace between stores costs a fee. One
question organizes everything: if each trace goes wherever its earned value best covers its costs, does
a sensible memory hierarchy assemble itself, and can we tell it apart from mechanisms whose persistence
profiles were designed rather than priced?

We take the position that the answer teaches more through its boundaries than its wins, so we registered
predictions in advance, ran the experiments, and report the failures with the same care as the hits.
Four contributions:

First, a formulation. Section 3 defines the objective and shows that exact caching, weight-decayed SGD,
EWC-style protection, and delta-rule decay all sit at corners of its tariff space. Designed mechanisms
are degenerate pricings; the interior of the space is the new territory.

Second, a mechanism and its economics. The greedy exchange rule is O(K) per step beyond SGD, and its
behavior splits into three regimes (saturated, selective, frozen) separated by switching costs. Through-
put responds to transition prices over a 175-fold range; maintenance prices never bind at tested scales.
What governs a priced router's behavior is the cost of moving knowledge, not the cost of holding it.

Third, controlled evidence with matched policies. Across stationary, drifting, volatile, and alternating
streams, value-based routing takes the best aggregate rank among capacity-limited competitors, and a
control with identical machinery but noise for valuation finishes last everywhere. The value signal
carries the effect. This replicates on a two-layer network.

Fourth, a map of our own failures. Bimodal residence times turned out to be structural, not priced.
Spike-locking does not exist at default settings, and chasing the null exposed a methodological trap:
a saturated controller's timing reflects its budget, not its policy, so any surprise-gating claim must
first show unsaturated operation. Reminiscence could not manifest by construction. And the neural-network
port killed our capacity-pressure hypothesis outright. Each failure narrows what any future claim about
economically routed memory is allowed to say.

We claim no victory over unlimited memory. Dense SGD retains more than every capacity-limited method
when retention is all that matters; we report it as a reference, not a rival. The subject here is what
happens below the ceiling.

## 2 Related Work

**Fast weights and multi-timescale synapses.** Hinton and Plaut (1987) paired fast, quickly-decaying
weights with slow ones. Ba et al. (2016) revived the idea for attention. Kaplanis et al. (2018) used
complex-valued synapses with task inference for continual reinforcement learning, and Benna and Fusi
(2016) derived cascade synapses whose continuum of timescales produces power-law forgetting. In each
case the persistence profile of a compartment is given a priori. Here profile assignment is the decision
variable.

**Memory-augmented networks and linear attention.** The NTM and DNC line (Graves et al., 2014, 2016)
learned addressing and free-list allocation inside one store. DeltaNet and gated delta-rule models learn
input-dependent writes into a decaying state (Schlag et al., 2021; Yang et al., 2024). Titans adds
surprise-gated test-time memory; TTT layers make the hidden state itself a trained model (Sun et al.,
2024). All of these decide what to write into a single medium. None prices movement across media against
maintenance.

**Nested optimization views.** Learned optimizers compress gradient history into update rules
(Veeriah et al., 2021). Nested Learning formalizes models as nested optimization problems and reads
optimizers as associative memories (Behrouz et al., 2025). That framework fixes the levels and their
context flows architecturally. Our tariffs answer a question NL leaves open: which memories belong at
which level, and at what price.

**Normative forgetting and consolidation.** Anderson and Schooler (1991) derived human retention curves
from need probability. Recent theory derives consolidation as compression for generalisation in latent
representations, and biological consolidation gates on recall (eLife RP 90793). Closest concurrent work
comes from two directions. Chen and Cheng (2026) learn a seven-factor value function for retaining
textual agent memories, sharing our normative question and our learned-valuation stance within a single
text store; no store multiplicity, maintenance tariffs, or movement prices appear there, and their value
factors are cognitive rather than predictive. Li et al. (2026) route facts between textual buffers and
model parameters with heuristic pipeline stages rather than a priced objective. Our scope differs on
three axes: any gradient learner rather than agents or latents alone, per-trace routing across the whole
update stack, and one objective with switching costs evaluated through pre-registered dynamical
signatures. KV-eviction schemes with importance scores (OBCache and successors) instantiate one-store
pricing for caches; we recover them as the same corner case as exact caching.

**Continual learning.** Regularization (Kirkpatrick et al., 2017; Zenke et al., 2017), replay, and
architectural methods manage forgetting; plasticity-loss results (Dohare et al., 2024) show retention
alone is not enough. PES reads both pathologies as mispricing: protection is an over-tariffed migration,
forgetting is under-tariffed decay, and plasticity loss is stale traces occupying capacity without paying
rent.

## 3 The Principle

### 3.1 Stores, traces, and prices

A learner sees a stream $(x_t, y_t)$ and maintains $K$ traces $w_i$. Prediction uses only resident
traces. Stores differ in three numbers: capacity $M_s$, a maintenance tariff $\lambda_s$ per step per
resident trace, and an unattended decay rate $\mu_s$. Fast storage is cheap, roomy, and leaky; slow
storage is expensive per step, protected, and scarce. Nothing else distinguishes them.

### 3.2 Value

Each trace carries a running estimate of its responsibility for prediction error. For the linear
predictor used in most experiments, trace $i$'s instantaneous responsibility is $|x_{t,i}(y_t -
\hat y_t)|$, smoothed by an EMA with rate $\rho$. This is deliberately cheap. Section 6 returns to what
the cheapness costs.

### 3.3 The routing objective and the greedy rule

Placing trace $i$ in store $s$ scores

$$\Pi(i,s) = v_i - \lambda_s - \mu_s \cdot \text{staleness}_s(i),$$

and moving $i$ between stores charges a one-time fee $\kappa$ against $v_i$, with a hysteresis window
between swaps. Three rules follow: admit stimulated newcomers to fast storage, evicting the worst-scoring
resident if needed; exchange fast-for-slow whenever the best candidate's slow score beats the worst slow
resident's fast score by more than $2\kappa$; and delete fully decayed traces.

**Proposition 1 (static optimality).** *Under additive-separable scores and capacity constraints, any
feasible placement not assigning each store its highest-scoring traces admits a single swap strictly
increasing total score; hence top-$M$ routing (and minimum-score eviction under pressure) is optimal.*
Proof and the interchange argument are given in Appendix A; the constraint structure is a transversal
matroid pair, so single-exchange hill-climbing reaches the optimum.

**Proposition 2 (dynamic myopia with fees).** *Against an arbitrary value sequence, with at most one
exchange per window opening, the fee-gated max–min swap rule is exactly optimal among one-move policies:
no policy can condition on an unpredictable future, so the myopic maximizer of realized gain minus
$2\kappa$ is optimal, and separability reduces it to best-donor/best-evictee form.* Proof in Appendix A.

Both statements allocate traces **given** prices; neither says the prices are honest. Three explicit
boundaries: synergistic value (non-additive $\Pi$) breaks Proposition 1; predictable value dynamics let
budget-saving look-ahead beat Proposition 2's myopia; and mispriced credit through a nonlinearity
collapses the advantage structure empirically (Section 5.5). The theorems cover allocation, not price
discovery — that division of labor is deliberate and load-bearing for the paper's honesty.

### 3.4 Known mechanisms as corner pricings

| Mechanism | Recovered by |
|---|---|
| Exact cache / KV | $M_{fast} \to \infty$, $\lambda_{slow} \to \infty$: never promote, never forget |
| Weight-decayed SGD | one store, uniform tariff |
| EWC / SI protection | $\lambda_{slow} \to \infty$ for protected coordinates |
| Delta-rule decay | one store with decay, ungated writes |
| Scheduled consolidation | exchanges on a clock instead of on scores |

Designed hierarchies are boundary points. The priced interior between them is what nobody has been
allocating.

## 4 Experiments

Streams come from a sparse linear regression with heterogeneous feature frequencies (common coordinates
appear often, rare ones rarely) and Poisson change-points that re-randomize part of the true support;
one regime alternates deterministically between two frozen rules. Capacity-limited learners share the
same predictor, the same SGD updates, the same capacities (45 fast + 15 slow), and the same value EMA.
They differ only in routing policy: PES; random-routing (identical machinery, noise for valuation);
clock-value and clock-magnitude (fixed-period swaps ranked by value or magnitude); single-store decay.
Dense SGD appears as an upper reference. Ten seeds per cell; paired sign tests; predictions P1–P5 and
competing hypotheses C-a/C-c were written down before the run. The neural port replaces the predictor
with a two-layer MLP over stably slot-bound inputs, calibrated for online convergence (learning rate
0.003 with clipping; details in the artifact log).

## 5 Results

### 5.1 Value-based routing organizes the aggregate

Among the five capacity-limited policies across four regimes, PES takes the best mean rank. The
random-routing control finishes last in every regime, three to five times worse than valued policies in
the network experiments. Since the control shares everything except the valuation signal, sparsity and
machinery cannot explain the ranking; the value signal does. Against dense SGD the story flips, as it
should: with no capacity limit, dense retention wins everywhere (0.0031 vs 0.1606 stationary tail).
Priced memory is a way to spend a budget well, not a way around having one.

Per-regime differences among valued policies stay within noise almost everywhere. One exception:
under deterministic alternation PES beats magnitude-clock (d = +0.94, p = 0.021). Under pure stationary
retention, random-routing significantly beats PES (d = −1.44, p = 0.002); when nothing changes, careful
routing buys nothing and its turnover costs a little. We did not expect that sign, and it survived.

### 5.1b Seed expansion and a second environment family

Re-running four policies at ten fresh seeds on the original Bernoulli streams and on a new
Gaussian-featured family (support-gated dense activations, higher effective capacity pressure)
sharpened the picture. Under drift, PES now beats every competitor significantly in both families:
vs fixed-clock 9/10 paired seeds (p = 0.021, mean gap −0.12 Gaussian, −0.03 Bernoulli); vs
single-store decay 10/10 (p = 0.002); vs noise-routing 9–10/10. The Gaussian family also exposes the
clearest capacity-pressure effect in the study: with every support coordinate active each step,
unstructured decay collapses (stationary tail 0.48 vs PES 0.16). In stationary cells PES ties clock
within noise and decisively beats decay and noise-routing (10/10, p = 0.002). One instability is
reported as found: the original run's significant stationary loss to noise-routing did not replicate
on fresh seeds (the reversal went 10/10 the other way), so that cell is treated as unresolved
variance rather than evidence either way. The stable, cross-family claim is: **value-priced routing
pays when the world changes, costs approximately nothing when it does not, and both failure modes it
protects against — unpriced decay and unpriced turnover — are real.**

### 5.2 Movement is the binding price

Sweeping maintenance tariffs across two orders of magnitude ($\lambda_{slow}$ from $10^{-5}$ to
$10^{-2}$, $\mu_{fast}$ from 0.005 to 0.08) leaves promotion throughput bit-for-bit identical, 0.07487
promotions per step in all twelve cells. Keeping-prices do not bind at operational scales. Transition
prices bind hard: widening the hysteresis window from 1 to 25 to 100 steps drives throughput from 1.92
to 0.068 to 0.011 promotions per step, a 175-fold range, and raising the migration fee κ from 0 to 0.2
cuts throughput monotonically until saturation takes over. If you want to control a priced router, tax
movement.

### 5.3 Three operating regimes

The dose-response surface splits behavior into bands. Below a switching-price threshold the router is
saturated: it moves something every time the window opens, so timing carries no information and only the
choice of which trace moves correlates with errors (about 1.15× baseline error just before swaps). At
moderate prices the router goes selective: throughput drops below budget. Above the threshold the router
freezes; rare migrations anti-correlate with errors (≈0.52). Whether selective-band timing tracks
errors was tested at n = 10 unsaturated seeds after preregistering a 1.05 contrast threshold: median
0.97, four seeds above threshold, one-sided p = 0.83 — **not supported**. The earlier suggestive mean
(1.17 at n = 3) did not survive power. The settled statement: this mechanism's migration *timing* is
set by price crossings against switching costs, not by prediction-error spikes, in every regime we can
test; priced persistence and surprise-gated memory are different control structures, and conflating
them is unsafe in either direction.

### 5.4 Registered signatures that failed

Three predictions failed cleanly. Residence-time bimodality showed up in every two-store method
(coefficient ≈ 0.73 for both PES and clock), so it marks the bookkeeping, not the pricing. Spike-locking
was absent at defaults, and the chase exposed the trap named above: two different mechanisms produce the
same null, and only a differential experiment separates them. Reminiscence could not manifest at all,
because fifteen slow slots cannot span twenty-five support coordinates per rule set; nothing survives a
visit. Each failure converts into a design constraint for the next attempt.

### 5.5 The network port replicated the control and killed our follow-up

With the MLP substrate, the necessity result held: noise valuation is catastrophic, valued policies
cluster tightly. But our registered capacity-pressure prediction inverted. Tightening capacity from 70
slots against 10 true supports down to 20 slots made fixed-clock consolidation better relative to greedy
exchange, not worse (stationary tails 0.013 vs 0.100 at 20 slots). The diagnosis: pressure raises routing
frequency exactly where first-order prices are least trustworthy. Through a tanh layer, ∂ŷ/∂x misses
interactions, so pressured exchanges evict valuable traces on misattributed credit, while a slow clock
samples the noisy price rarely and averages the noise away. Pricing needs a price worth trusting; naive
gradient tags degrade fastest in busy markets.

Ablations on the linear side agree with the mechanism reading: removing the switching charge, the
hysteresis window, the fast decay, or the slow tariff each costs accuracy (volatile tails 0.240 → 0.270,
0.274, 0.305, 0.268 respectively), and removing the valuation signal via the random control costs far
more (0.284 with wider spread).

## 6 Discussion

The results support a narrow but real thesis. Value-priced persistence is necessary for organized memory
under scarcity: take the signal away and the same machinery collapses. It is not sufficient: our cheapest
valuation works at linear scale and loses to a patient clock on networks under pressure, which locates
the open problem precisely. Influence-style re-scoring, leave-one-out probes at decision moments, learned
critics, and longer integration windows are the obvious next attempts, and R6's inversion tells us
exactly where to aim: the regime where routing frequency outruns price reliability.

Two methodological points generalize past this project. First, check whether a controller's action budget
is saturated before interpreting its timing; several published gating mechanisms may be describing their
clocks. Second, negative results with diagnoses carry more design information than marginal positive
results; the failures in Section 5.4 constrain future claims more than the aggregate win in 5.1.

Limitations are real. Linear scale for most cells, three seeds on the network port, greedy routing
without synergy guarantees, first-order value attribution, one environment family, and no deep-network
deployment. The signatures remain undemonstrated at scale. This is a framework paper with laws measured
in vitro, not a system paper.

## 7 Conclusion

Memory hierarchy structure can be derived rather than designed. One priced allocation objective organizes
traces across persistence media, recovers familiar mechanisms as corners of its tariff space, wins the
aggregate among matched policies through its value signal alone, obeys a measurable law in which movement
prices dominate keeping prices, and fails in informative ways when its prices are untrustworthy. What a
learning machine should remember is, to a first approximation, whatever earns its keep minus what it costs
to keep; making the accounting honest under nonlinearity is the problem we leave open, on purpose, with
the map of where we already fell.

## Reproducibility statement

All code runs on numpy plus matplotlib, with torch only for Section 5.5. Twenty-three unit tests gate
every change; environment seeds derive from crc32 of regime names and are stable across machines; every
number above exists in committed JSON artifacts with generation scripts; predictions and competing
hypotheses were registered in tracked notes before the corresponding runs. Full artifacts accompany the
submission.

## References (to be formatted)

Hinton & Plaut (1987); Ba et al. (2016); Schlag et al. (2021); Yang et al. (2024); Behrouz et al. (2024)
Titans; Sun et al. (2024) TTT; Behrouz et al. (2025) Nested Learning; Graves et al. (2014, 2016);
Veeriah et al. (2021); Kirkpatrick et al. (2017) EWC; Zenke et al. (2017) SI; Dohare et al. (2024)
plasticity loss; Anderson & Schooler (1991); Benna & Fusi (2016); Kaplanis et al. (2018); eLife RP 90793;
Li et al. (2026a, arXiv:2608.22215) dual-layer agentic memory; Chen & Cheng (2026, arXiv:2606.12945)
Learning What to Remember; Li (2026b, arXiv:2604.18002) Neural Garbage Collection; OBCache
(arXiv:2510.07651). [Full bibliographic entries in research/literature/FIELD_MAP.md and SWEEP5_FINAL.md.]

## Appendix A — Proofs of Propositions 1 and 2

See theory/THEOREMS.md in the artifact bundle (reproduced here in camera-ready). Proposition 1: per-trace
decomposition plus identical slots within a store reduces feasibility to occupancy counts; within a store,
any non-top-$M$ assignment loses to a within-store swap; across stores, unsorted score margins admit an
improving cross-store exchange, and connectedness of the exchange graph under the transversal-matroid
interchange axiom delivers global optimality of sorted placements. Corollary: unique eviction target.
Proposition 2: at an opening, the action space is {hold} ∪ {one pair swap}; realized change from swapping
$(i\to s, j\to f)$ is $\Pi_t(i,s)+\Pi_t(j,f)-\Pi_t(i,f)-\Pi_t(j,s)$ less fees charged to future scoring
through reduced value; arbitrariness of the post-opening sequence makes myopia optimal within the class,
separability picks donor and evictee independently, and hold dominates negative-gain swaps. Boundaries:
additivity, unpredictability, and price honesty respectively; Section 5.5 exhibits the empirical
consequence of violating the third.

---

## IEEE adaptation note

For TNNLS/TAI: two-column format; move §3.4 table into §5; expand the complexity analysis in §3.3 into a
numbered proposition with proof sketch; convert references to IEEE numeric style. The abstract's fourth
sentence should be shortened and the failure paragraph moved to the introduction's contribution list,
where IEEE reviewers expect limitations up front.
