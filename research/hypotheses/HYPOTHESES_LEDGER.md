# HYPOTHESES LEDGER — 52 Candidates (v1)

Scoring: Cn=conceptual nov, Mn=math nov, Ep=empirical potential, Gn=generality, Si=simplicity,
Td=theoretical depth, Pr=practical import, Fe=feasibility, Pa=prior-art distance, Im=paradigm impact.
Weights: (.15,.10,.10,.12,.07,.08,.12,.11,.10,.05). Level: L1 engineering … L6 paradigm.

## FAMILY A — PERSISTENCE/MEMORY ABSTRACTION

### H01 — Persistence-Routed Learning ("Memory Economics") 
Principle: Every trainable tensor (activations, optimizer moments, weights) is one medium in a shared
persistence space; a single objective trades predictive value against maintenance+capacity costs, and its
optimal solution IS the routing of information between media. No fixed β, no hand-set consolidation rate.
Challenges: fixed store semantics; train/inference boundary; consolidation schedules.
Math: min_{routing π} E[L(pred)] + Σ_s λ_s·C_s(state_s) s.t. capacity Σ_s M_s ≤ B; traces carry value v_i,
cost c_i(s)=maintenance+interference; optimal π* assigns trace i to argmin store of amortized
(c_i(s)+λ_s m_i)/v_i; gradient flow emerges as continuous limit of π*.
Computation: each trace has scalar bookkeeping (value, age, interference); periodic re-routing moves
entries between activation/moment/weight tensors.
Why matter: could unify context length, optimizer design, continual learning, and consolidation under one
law; predicts when inference-time learning should persist.
Prior: Nested Learning (fixed hierarchy); Titans (hand gating); predictive-forgetting theory (latents only);
HiPPO (fixed bases); Hinton&Plaut (fixed decay).
Difference: routing itself is derived from one priced objective spanning ALL media incl. activations;
special cases recover Adam (no promotion), KV-cache (exact fast store), EWC (infinite maintenance cost).
Falsifiable prediction: optimal routing becomes bimodal (keep-vs-promote) as load increases; promotion
events cluster after prediction-error spikes; removing routing degrades specifically long-range recall.
Minimal experiment: online regression w/ drifting rules + rare repeated items; compare routed 3-store vs
matched single-store/fixed-decay baselines on regret + recall of rare items. Level L5.
Score: 9,7,8,9,7,8,8,8,8,9 → **8.35**

### H02 — Trace-Valence Synapses
Per-weight "survival curve" parameters (not just magnitude): each synapse stores a distribution over
persistence half-lives; updates shift mass rather than overwrite values. Challenges: flat weight semantics.
Math: θ_ij(t+Δ)=Σ_k p_k(t)·θ^k_ij, p evolves by reward/error likelihood; memory = mixture over ages.
Prior: SI, complex synapses (Kaplanis), cascade models (Benna–Fusi).
Difference: cascades are fixed kernels; here kernel shape learned by task statistics.
Prediction: learned half-life spectra match environmental change-point statistics.
Experiment: synthetic drift tasks w/ known change rates; check spectrum tracks them. L3.
Score: 7,6,7,6,6,6,6,8,6,5 → 6.35

### H03 — Maintenance-Cost Regularization
Learning = cache management: add explicit maintenance term μ‖∂state/∂t‖ to loss; forgetting emerges from
budget, not decay. Math: L+μ∫‖ṡ‖dt; Euler–Lagrange gives elastic consolidation dynamics.
Prior: weight decay (parameter-norm only), IB (information-level).
Difference: cost on STATE VELOCITY across all stores, not norm or MI.
Prediction: μ sweeps reproduce grokking-like delayed transitions.
Experiment: two-layer net on algorithmic task; sweep μ; look for memorize→generalize transitions. L3.
Score: 6,6,7,6,7,6,5,8,6,4 → 6.0

### H04 — Forgetting-as-Inference
Forgetting recast as approximate Bayesian model selection over traces (evidence-weighted deletion).
Math: keep trace iff log p(data|trace) > threshold − Occam factor; delete otherwise.
Prior: predictive forgetting (2603.04688), SNL pruning.
Difference: per-trace evidence accounting during ONLINE operation, not offline compression stage.
Prediction: deleted traces are those with high description length AND low reuse; matches spacing effect.
Experiment: associative memory w/ correlated items; measure which get forgotten vs MDL ranking. L3.
Score: 6,6,6,6,6,7,5,7,6,4 → 5.85

### H05 — Interference-Priced Writes
Write decisions priced by PREDICTED interference with existing contents (crosstalk forecast).
Math: accept write w iff Δutility > κ·Σ_j sim(w,m_j)·v_j; rejection routes to slower store.
Prior: OGD/GPM (gradient projection post-hoc); Hopfield capacity analyses.
Difference: interference enters the WRITE decision, not the update direction.
Prediction: throughput-recall Pareto frontier dominates delta-rule at equal capacity.
Experiment: key-value recall at increasing load; compare acceptance-policy vs unconditional writes. L2-3.
Score: 6,6,7,6,6,5,6,8,6,4 → 5.95

### H06 — Store-Topology Learning
Number/kind of stores itself learned (neural architecture search over persistence structures).
Math: DAG over store nodes; edges=transfer ops; searched by validation regret.
Prior: NAS (compute-focused), Dynamic Nested Hierarchies (arXiv 2511.14823).
Difference: search space defined purely by persistence/cost axes, not layer types.
Prediction: discovered topologies recapitulate hippocampus-cortex-like splits on drift tasks.
Experiment: tiny NAS over store graphs on synthetic lifelong streams. L2.
Score: 5,5,6,6,4,4,6,5,5,5 → 5.15

### H07 — Consolidation-by-Utilization
Promotion probability ∝ retrieval frequency² (spacing-effect law made mechanical); unused fast traces die.
Math: P(promote i)=σ(a·u_i+b·u_i²−c·age_i); u=retrieval count.
Prior: psychological spacing literature (row 48); ACT-R activation.
Difference: no ML work derives promotion from utilization statistics within one optimization objective.
Prediction: optimal schedule emerges matching human spacing gaps given same retention tests.
Experiment: replay-based learner; compare utilization-triggered promotion vs random/uniform on retention. L2.
Score: 6,5,7,5,8,5,6,8,7,4 → 6.05

### H08 — Cross-Store Gossip
Stores act as mutual teachers: activations distill into moments, moments into weights, weights into
architecture stats — asynchronous gossip convergence to shared knowledge.
Math: s_k ← (1−α)s_k + α·D_k(s_{k−1}); D=distillation map; consensus ⇒ robust knowledge.
Prior: mean teacher, self-distillation, deep monocular distillation chains.
Difference: bidirectional cyclic gossip with convergence guarantees, not one-way EMA.
Prediction: cyclic version beats any acyclic chain at equal compute under noise injection.
Experiment: 3-store toy; inject label noise; compare cycle vs chain vs none. L2.
Score: 5,5,6,5,6,5,5,7,5,4 → 5.3

## FAMILY B — OPTIMIZATION REIMAGINED

### H09 — Optimizer-State-as-Knowledge
Deploy WITH optimizer state: moments carry compressed task info that should inform downstream tasks;
formalize moment-transfer learning.
Math: m_t as sufficient statistics of past gradients ≈ Fisher information field; transfer m_0 for task B.
Prior: checkpointing practice; optimizer-state warm-start folklore; Nested Learning framing.
Difference: first controlled study + theory of WHEN moment transfer beats weight transfer.
Prediction: moment transfer wins under partial feature overlap; loses under full overlap (redundant).
Experiment: small CNNs/matrix factoration; cross-task moment init vs weight init grid. L2.
Score: 6,5,8,6,8,5,7,9,6,4 → 6.3

### H10 — Gradient-Trajectory Compression
Store compressed TRAJECTORY (low-rank path integral) not endpoint; adaptation = trajectory continuation.
Math: θ_T + U r where U spans principal path subspace; new-task updates constrained to span.
Prior: mode connectivity, weight interpolation, LoRA (output-space analog).
Difference: operates on optimization PATHS not weight deltas; predicts which edits are safe.
Prediction: low-rank path subspace generalizes across related tasks better than equal-rank weight deltas.
Experiment: train on A; extract path PCA; compare path-subspace vs delta-subspace fine-tuning on B. L2.
Score: 6,6,7,6,6,6,6,8,6,4 → 5.95

### H11 — Learned Timescale Spectra
Replace scalar β's with LEARNED spectral profile per parameter group; optimizer learns its own filter bank.
Math: m_t(ω) filters gradient frequencies; β(ω) parametrized, trained by meta-loss.
Prior: Muon spectral filtering analysis (2606.03899); AdaBelief etc. (diagnostic scalars).
Difference: spectrum is a learned function, target frequencies tied to task nonstationarity.
Prediction: learned spectra develop peaks at environmental change frequencies on drifting data.
Experiment: sinusoidally-drifting regression; inspect learned filter bank. L2-3.
Score: 6,6,7,6,6,6,6,8,5,4 → 5.85

### H12 — Optimizer as Amortized Posterior
Moments = variational posterior q(θ*|history); update rule = amortized inference; deployment uses samples.
Math: ELBO over θ* with q parametrized by moment network; SGD+momentum = mean-field special case.
Prior: BayesByBackprop, SWAG (endpoint covariance), learned optimizers.
Difference: posterior is OVER OPTIMUM conditioned on gradient history, not over weights directly.
Prediction: posterior samples beat point estimates OOD without explicit ensembling.
Experiment: quadratic bowls→shifted bowls; sample from moment-posterior vs point. L2.
Score: 6,7,6,6,5,7,5,7,6,4 → 5.75

### H13 — Prequential Coding Objective
Train to minimize ONLINE codelength of data AS IT ARRIVES (prequential score), not batch average loss.
Math: L_pre = Σ_t −log p(x_t|x_<t; θ_t) — the actual MDL of the stream; every regularizer = extra bits.
Prior: prequential analysis in statistics; MDL learning theory; never used as NN training objective.
Difference: changes THE OBJECTIVE, making train/test gap meaningless by construction.
Prediction: prequential-trained nets show smoother calibration and different scaling exponents.
Experiment: small LM on wiki corpus; prequential vs standard SGD; measure codelength+calibration. L3.
Score: 8,7,7,8,7,7,7,7,8,6 → **7.3**

### H14 — Regret-Matched Parameter Updates
Parameter perturbations maintained as experts; Hedge/multiplicative weights allocates across them.
Math: θ=Σ_i w_i θ^i; w_i ∝ exp(−η Σ losses of perturbation i); regret bounds transfer from OCO.
Prior: ES population methods, bandit superoptimizers; not applied as PRIMARY update law for NN params.
Difference: theoretical regret guarantees inherited; no gradients needed for the mixing layer.
Prediction: competitive with SGD on noisy/nonstationary objectives where gradients mislead.
Experiment: adversarial-quadratic + drifting MNIST head; hedge-mixing vs Adam. L2.
Score: 6,6,6,5,6,7,5,7,6,4 → 5.65

### H15 — Bilevel Self-Distillation Stack
Explicit stack where level k trains level k+1 by distilling its OWN dynamics (not outputs); each level =
slower copy. Math: min_{θ_{k+1}} KL(dyn_{θ_k} || dyn_{θ_{k+1}}) over rollout distribution.
Prior: Nested Learning Hope; Born-again networks.
Difference: distills PROCESS (update dynamics) not predictions; stack depth adaptive to nonstationarity.
Prediction: deeper stacks help exactly when environment has multi-rate structure; measurable via matched spectra.
Experiment: two-rate drift env; vary stack depth; check alignment to true rates. L2-3.
Score: 6,6,6,7,5,6,6,7,5,5 → 5.9

## FAMILY C — TRAINING/INFERENCE UNIFICATION

### H16 — Inference-as-Learning Ledger
Every inference episode appends priced ledger entries; system optimizes lifetime regret, not per-batch loss.
Math: θ_{t+1}=θ_t+π(entry features; budget); π trained to minimize discounted future loss.
Prior: TTT/TTA (ephemeral), test-time nearest-neighbor training, continual TTA surveys.
Difference: writes are permanent, priced, and POLICY-controlled across episodes — closing boundary entirely.
Prediction: positive ledger policies show reminiscence effect (performance bump on re-visits exceeding fit).
Experiment: stream of related tasks w/ revisits; compare ledger policy vs frozen vs naive-accumulate. L2.
Score: 8,7,8,8,7,7,8,7,7,7 → **7.55**

### H17 — Test-Time Consolidation Passes
After each episode, brief unsupervised "sleep" consolidates episode traces into weights via replay.
Math: replay z~q(episode); θ←θ−∇[L(z)+β KL(p(z‖θ_old)||p(z‖θ_new))].
Prior: generative replay (Shmelkov et al.), sleep-phase algorithms, GENESIS.
Difference: consolidation triggered/amount decided by measured episode VALUE (novelty×confidence).
Prediction: value-gated consolidation beats always-consolidate at equal total compute.
Experiment: episodic continual classification; gate by value estimate. L2.
Score: 6,5,7,7,7,5,7,8,5,4 → 6.05

### H18 — Prediction-Market Models
Subnetworks stake persistence-capital on predictions; correct bettors gain capital (capacity), wrong lose.
Math: w_i←w_i·exp(γ·r_i·stake_i); capital determines parameter share; bankruptcy prunes.
Prior: boosting, mixture-of-experts, hedge updating (H14 kin).
Difference: stakes couple PERSISTENCE (capacity allocation) to accuracy — an economy over subnets.
Prediction: spontaneous expert emergence WITHOUT explicit routing losses; specialization matches latent factors.
Experiment: multi-domain stream; observe capital/specialization co-emergence vs MoE baseline. L2-3.
Score: 7,6,7,7,6,5,6,7,6,5 → 6.3

### H19 — Self-Labeling Consolidation
Model's own high-confidence predictions on unlabeled deployment data become slow-store training targets
(weighted by calibrated confidence).
Math: pseudo-label weight = calibrated confidence^τ × novelty; thresholded writes.
Prior: self-training/pseudo-labeling (transductive), Noisy Student.
Difference: labels enter a PERSISTENCE-WEIGHTED store with anti-drift safeguards; studied as continual principle.
Prediction: improves only when confidence calibration transfers; predictably harms under miscalibration.
Experiment: digits shift sequence (MNIST→MNIST-M…); track drift of consolidated labels. L2.
Score: 5,4,6,6,8,4,6,8,4,3 → 5.3

### H20 — Query-Conditional Memory Depth
Memory traversal depth allocated per query by uncertainty (like adaptive computation but over MEMORY).
Math: depth d(x)=ceil(f(H[p(y|x,k_{1:d})])); stop when entropy plateaus.
Prior: adaptive computation time, product-key memories, RAG-k hops.
Difference: uncertainty-stopping criterion over memory READS, unified across parametric+nonparametric.
Prediction: easy queries use shallow reads with no accuracy loss; hard queries self-depthen.
Experiment: QA w/ mixed difficulty; measure depth-accuracy-cost frontier vs fixed-depth. L2.
Score: 6,5,7,6,7,5,6,8,5,4 → 5.8

### H21 — Continual Pretraining via Episodic Pricing
Assign each training token a price from current model state (surprise×future utility estimate); curriculum
emerges from the price field.
Math: price τ_t=g(loss_t, grad_norm, rarity); sampling ∝ price; budget-constrained.
Prior: curriculum/active learning heuristics, RHO-loss (prioritizes by irreducible loss).
Difference: prices feed a persistent BUDGET mechanism coupling data order to store states.
Prediction: budget-priced ordering beats RHO on multi-epoch corpora (accounts saturation).
Experiment: char-LM scale; compare orderings under fixed token budget. L3.
Score: 6,5,7,6,6,5,7,7,5,4 → 5.8

## FAMILY D — LEARNING ALGORITHMS/SELF-MODIFICATION

### H22 — Algorithm-Carrying States
Networks learn to carry executable ALGORITHMS as persistent dynamical objects (attractor manifolds
implementing procedures), separable from content.
Math: state decomposes x=c⊕a; a∈algorithm manifold; dynamics preserve a while c updates.
Prior: neural execution/interpretability (trained then analyzed); Neural Programming via searches.
Difference: algorithm manifold enforced during TRAINING as persistent object; transferable across domains.
Prediction: algorithm component transfers zero-shot to new symbol vocabularies.
Experiment: train sort/parity nets w/ decomposition penalty; swap input embeddings across tasks. L3.
Score: 7,7,6,6,5,7,6,5,7,5 → 6.35

### H23 — Executable Memory
Memory entries are PROGRAMS (tiny parameterized functions), not vectors; recall executes.
Math: M={(π_i, args_i)}; retrieve k* = argmax sim(q, meta(π_i)); output π_{k*}(q,args).
Prior: memory networks (vector slots), NTM (differentiable addressing of vectors), program synthesis.
Difference: storage unit = function; composes and generalizes where vector recall interpolates.
Prediction: compositional generalization jumps where vector-memory fails at equal capacity.
Experiment: SCAN-like compositional split; program-memory vs NTM baseline. L3.
Score: 7,7,6,6,5,6,6,5,7,5 → 6.05

### H24 — Proof-Carrying Weights
Knowledge stored with derivation metadata enabling verified editing (edit = replace derivation step).
Math: each fact f has proof tree T_f in learned logic; edits patch T_f; consistency checked by propagation.
Prior: model editing (ROME/MEMIT lack derivations); neurosymbolic KBs (external logic).
Difference: proofs live INSIDE the network as auxiliary heads; editing verified internally.
Prediction: ripple errors reduced vs MEMIT at equal edit count.
Experiment: GPT2-small on edited facts w/ ripple eval; proof-head aux loss. L3-4 (needs LM).
Score: 6,6,6,5,4,6,6,4,6,4 → 5.35

### H25 — Self-Applying Update Operators
Update operator U trained on tasks INCLUDING "improve U" — recursion bounded by outer budget.
Math: θ* = U(θ,D); U' ← U(U-history); meta-objective includes U-improvement rewards.
Prior: Gödel machine (proof-based, impractical), SRWM (self-modification without operator learning), Viterma/VeLo.
Difference: U's improvement is a FIRST-CLASS supervised signal, not emergent or proof-gated.
Prediction: bounded self-application yields monotone inner-loop speedups for K steps then plateaus safely.
Experiment: learned optimizer improving itself on quadratic family; measure safe plateau. L3.
Score: 7,8,5,6,4,8,5,4,8,7 → 6.35

### H26 — Universal Update Machines
Update rule itself Turing-complete neural module trained across diverse RL/online tasks to BECOME a
general learning algorithm (beyond meta-optimizer scope).
Math: s_{t+1}, θ_{t+1} = Φ(s_t, θ_t, o_t, r_t); Φ recurrent, trained by truncated meta-gradients.
Prior: learned optimizers (Veeriah), SRWM, AFSPO variants.
Difference: targets GENERAL learning behavior incl. memory formation strategies, not just descent steps.
Prediction: transfers to novel task families where learned optimizers fail (documented failure regime).
Experiment: meta-train on gridworlds; zero-shot to bandit/sequence tasks; vs VeLo/Adam. L3.
Score: 6,6,6,7,4,6,6,4,6,5 → 5.65

## FAMILY E — DYNAMICS/DYNAMICAL SYSTEMS

### H27 — Learned Bifurcation Control
Training shapes the bifurcation diagram of network dynamics; learning = pushing system near criticality
for max sensitivity w/ stability basins for memory.
Math: control parameter ρ(x) learned; d x/dt = F(x,ρ); memory = multistability created deliberately.
Prior: criticality hypotheses in neuroscience; reservoir tuning (static).
Difference: bifurcation structure as TRAINING TARGET with differentiable continuation.
Prediction: trained systems sit near saddle-node pairs on held-out inputs; measurable by normal forms.
Experiment: RNN on working-memory tasks; estimate eigenvalue margins vs trained baseline. L3.
Score: 6,7,5,5,4,8,4,4,7,5 → 5.5

### H28 — Plastic-Timescale Reservoirs
Reservoir spectral placement ADAPTS online to input timescale statistics (echo-state property maintained
at edge).
Math: W scaled to keep ‖W‖=ρ*(input autocorr); ρ* tracked by spectral estimation.
Prior: ESN (random fixed), intrinsic plasticity rules (fixed homeostasis).
Difference: timescale MATCHING as explicit online objective with stability constraint.
Prediction: tracking reservoirs dominate fixed ones on multi-scale switching signals.
Experiment: Mackey-Glass w/ regime switches; adaptive vs fixed ρ. L2.
Score: 5,5,6,5,7,5,4,8,5,3 → 5.05

### H29 — Nonequilibrium Drive-Dissipation Learning
Memory formation modeled as balance of external drive (data) and dissipation (decay); learning rules
derived from entropy production, giving hardware-friendly local updates.
Math: dθ/dt=−∇_θ L − γθ + η(t); entropy production rate Σ bounds learning speed (Landauer-style).
Prior: equilibrium propagation (near-equilibrium), thermodynamic computing proposals.
Difference: operates FAR from equilibrium with dissipation as resource, not nuisance.
Prediction: optimal dissipation schedule exists; violating it causes characteristic error plateaus.
Experiment: physical-inspired simulation; sweep drive/dissipation ratio; map phase diagram. L2-3.
Score: 6,7,5,5,5,7,5,5,7,5 → 5.6

### H30 — Synchronization-Lock Learning
Knowledge = synchronized phase relations across network replicas; learning adjusts coupling until lock-in.
Math: Kuramoto-style dφ_i/dt=ω_i+Σ K_ij sin(φ_j−φ_i); memory = locked clusters; K trained.
Prior: oscillator networks (analysis); Hopfield (energy view); sync applied to segmentation only.
Difference: STORES information in phase relations with trained coupling — orthogonal to weight-content.
Prediction: capacity scales with coupling graph structure; interference shows phase-reset signatures.
Experiment: associative memory via Kuramoto net; compare capacity vs Hopfield same-size. L3.
Score: 6,7,5,4,5,6,4,6,7,4 → 5.3

### H31 — Hamiltonian Weight Dynamics
Fast weights follow conservative (Hamiltonian) dynamics exploring solution manifold; slow weights damp
into accepted basins — two physics regimes, one system.
Math: {θ,q} symplectic + Rayleigh damping on θ only; acceptance by Metropolis on slow clock.
Prior: HMC sampling (offline), underdamped Langevin, symplectic nets.
Difference: persistent exploration-damping SPLIT as architectural principle for continual settings.
Prediction: escapes sharp minima without noise tuning; flat-minima preference emerges naturally.
Experiment: loss landscape probes; compare basin sharpness vs SGD/Adam/SAM. L2-3.
Score: 6,7,5,5,5,7,5,6,6,4 → 5.6

## FAMILY F — INFORMATION-THEORETIC

### H32 — Rate-Distortion Lifelong Allocation
Global RD optimization allocates bits ACROSS stores and time; forgetting = optimal distortion increase.
Math: min Σ_t R_t s.t. D_t≤ε_t; Lagrangian yields per-trace bit budgets; ε_t tracks environment entropy.
Prior: IB deep learning debates; MDL online (H13 kin); predictive-forgetting theory.
Difference: explicit MULTI-STORE bit allocation over TIME, not single representation bottleneck.
Prediction: bit budgets shift toward slow stores as environment stabilizes — measurable.
Experiment: drift env w/ known entropy; track allocated bits per store vs true entropy. L3.
Score: 7,8,7,8,6,8,6,6,7,6 → **7.0**

### H33 — Bottleneck Cascade
Each store is an IB bottleneck of the previous: activations→compress→moments→compress→weights.
Math: min I(S_k;S_{k-1}) − β_k I(S_k;Y); β_k increasing down-stack.
Prior: IB (single bottleneck); information plane analyses.
Difference: CHAINED bottlenecks define the whole memory hierarchy; β's learned.
Prediction: optimal β ratios predictable from layer receptive-field sizes.
Experiment: conv nets; measure MI planes per cascade level; compare vs random β. L3.
Score: 6,7,6,6,5,7,5,6,5,4 → 5.7

### H34 — Predictive-Information Criterion
Keep whatever maximally reduces FUTURE surprise: persistence value = predictive information I(past→future).
Math: v_i = I(trace_i ; future observations); greedy keeps top-v traces under budget.
Prior: predictive information bottleneck (Bialek), empowerment (RL), surprise-based memory (Titans kin).
Difference: applied as TRACE-LEVEL routing statistic across stores, computed online cheaply.
Prediction: v_i estimates rank-correlate with counterfactual removal damage ≥0.7.
Experiment: instrumented learner; correlate estimated v_i with actual removal impact. L2.
Score: 7,6,8,8,7,6,6,7,6,5 → 6.75

### H35 — Compression-Progress Curriculum
Attention/training focused where COMPRESSION GAIN is currently highest (learning progress as signal).
Math: sample weight ∫ Δ(compression achieved)/Δt; intrinsic-motivation transplant.
Prior: Oudeyer-style intrinsic motivation (RL); RHO loss (irreducible loss proxy).
Difference: compression PROGRESS (second derivative) drives allocation, not loss level.
Prediction: focuses on learnable-but-unlearned regions; avoids both easy and impossible data automatically.
Experiment: synthetic mixture of trivial/noisy/instructable clusters; verify selection dynamics. L2.
Score: 5,5,7,6,7,5,5,7,5,3 → 5.45

### H36 — Entropy-Budgeted Plasticity
Each layer gets plasticity budget proportional to its output entropy production; stale layers freeze
gracefully (demote) instead of dying.
Math: g_l = clamp(H(out_l)/B_l); gradients scaled by g_l; budgets adapt.
Prior: plasticity-loss mitigations (continual backprop, resets); dormancy metrics.
Difference: entropy-linked SOFT demotion preserving knowledge while restoring flexibility.
Prediction: prevents unit death without discarding knowledge (unlike resets); measurable dormancy curves.
Experiment: long continual streams; compare vs continual backprop/resets on plasticity+retention. L2-3.
Score: 6,5,7,6,7,5,7,8,5,4 → 5.95

## FAMILY G — STATISTICAL/BAYESIAN

### H37 — Persistent Particle Filter over Parameters
Multiple parameter particles at annealing temperatures persist across tasks; resampling = consolidation.
Math: p(θ|D_1:t)≈Σ w^i δ(θ−θ^i); temperature ladder; resample by cumulative evidence.
Prior: sequential Monte Carlo, particle-based meta-learning, SWAG (Gaussian approx only).
Difference: persistent multi-modal posterior ACROSS task boundaries with principled resampling.
Prediction: escapes single-mode traps in multi-modal loss landscapes where SGD restarts fail.
Experiment: bimodal synthetic posteriors across task sequences; mode coverage vs Adam+restarts. L2.
Score: 6,6,6,6,4,7,5,6,6,4 → 5.6

### H38 — Volatility-Gated Multi-Store Kalman
State-space model where Kalman gains route measurements into fast/slow stores by inferred volatility.
Math: two-state x_f,x_s; gain K_vol=σ(volatility estimate); volatility learned by changepoint model.
Prior: adaptive Kalman filtering; behavioral volatility-learning models (PMC7329063).
Difference: multi-store extension with LEARNED routing — connects behavioral math to NN memory stacks.
Prediction: matches or beats optimal filtering on simulated volatile environments; gains show threshold behavior.
Experiment: volatile regression; compare to oracle-gain filters. L2.
Score: 6,6,7,6,6,6,5,7,5,4 → 5.8

### H39 — Thompson-Sampling Consolidation
Promotion decisions sampled from posterior over "will this be useful"; exploration in memory management.
Math: promote w.p. P(u_i>τ|data); posterior updated by realized reuse counts.
Prior: Thompson sampling (bandits); not applied to memory routing decisions.
Difference: treats CONSOLIDATION as sequential decision problem with explicit exploration bonus.
Prediction: discovers non-obvious useful traces (exploration wins) vs greedy value ranking.
Experiment: delayed-usefulness environments (traces useful only later); TS vs greedy promotion. L2.
Score: 6,5,7,6,7,5,5,7,6,4 → 5.8

### H40 — Changepoint-Driven Demotion
Bayesian online changepoint detection DEMOTES (not deletes) pre-change knowledge to cold storage.
Math: BOCPD on loss stream; hazard → demote segment-specific traces; restore on reverse-shift.
Prior: BOCPD (filtering); continual learning ignores changepoints almost entirely.
Difference: temporal STRUCTURE of drift controls memory organization; reversible demotion.
Prediction: handles alternating environments (A/B/A/B) near-perfectly where CL baselines catastrophically fail.
Experiment: alternating domain streams; demotion vs EWC/replay. L2.
Score: 7,6,8,7,7,6,7,8,6,5 → **6.75**

## FAMILY H — NEUROCOMPUTATIONAL PRINCIPLES

### H41 — Cascade-Synapse Deep Learning
Benna–Fusi synaptic cascade (continuous timescale continuum in each synapse) transplanted to DL with
learned readout of cascade state.
Math: θ = Σ_k α_k c_k; dc_k/dt = −c_k/τ_k + f(c_{k+1}) − g(c_{k−1}) + input; α learned.
Prior: Benna–Fusi theory (neuro); Kaplanis complex synapses (coarse 2-compartment).
Difference: FULL cascade with learned readout in deep nets; power-law forgetting emerges, not imposed.
Prediction: retention curves show power-law scaling matching biological data; outperforms exponential decay.
Experiment: continual image streams; measure forgetting curves vs exponential baselines. L2-3.
Score: 7,6,7,6,6,6,6,7,6,5 → 6.3

### H42 — Learned Global Modulators
Small set of global scalars (dopamine/ACH/noradrenaline analogs) CONTROL routing between stores; modulator
policy learned end-to-end.
Math: m_t=π(summary stats); updates gated by m_t: Δs_k∝m_k,t·δ_t.
Prior: neuromodulated plasticity (fixed rules), meta-learned neuromodulation (Miconi - RNN only).
Difference: modulators control CROSS-STORE ROUTING policy in deep learners, trained by meta-gradients.
Prediction: distinct modulator roles emerge (novelty vs reward vs volatility) matching biological dissociations.
Experiment: 3-store learner; inspect learned modulator usage across designed task statistics. L2-3.
Score: 6,5,7,6,6,5,6,7,5,4 → 5.7

### H43 — Uncertainty-Driven Generative Replay
Replay generated WHERE the generative model is most uncertain (targeted dream), not uniformly.
Math: sample z~argmax H[p_θ(y|z)]; consolidate those; iterate.
Prior: generative replay (uniform), sleep-wake algorithms, HERSTORY variants.
Difference: replay allocation by model UNCERTAINTY — active dreaming.
Prediction: targeted replay dominates uniform at equal replay budget, especially low-budget regime.
Experiment: class-incremental w/ VAE replay; targeted vs uniform sampling curves. L2.
Score: 6,5,7,6,7,5,6,8,5,4 → 5.95

### H44 — Two-Process Eligibility Architecture
Fast process (eligibility, per Hebbian traces) proposes; slow process (consolidation controller) disposes.
Math: e_t decays fast; θ̇=e_t·g(e-history); g learned gate deciding which eligible traces commit.
Prior: eligibility traces (RL, fixed λ), three-factor rules (fixed third factor).
Difference: the COMMIT FUNCTION is learned — credit assignment across timescales is trained.
Prediction: learned g discovers delay windows matching task credit-assignment structure.
Experiment: delayed-reward tasks w/ varying delays; inspect g's effective window. L2.
Score: 6,6,6,6,6,5,5,7,5,4 → 5.55

## FAMILY I — GAME/ECONOMIC FORMS

### H45 — Parameter Markets
Parameters (or modules) TRADE capacity using internal pricing; high-value params buy budget, low-value sell.
Math: budget_i(t+1)=budget_i+r_i−p·usage_i; r=value estimate; constraints keep Σ fixed.
Prior: MoE (routing without economics), lottery tickets (post-hoc), soft pruning.
Difference: dynamic market-clearing DURING training; capacity follows marginal value continuously.
Prediction: emergent sparsity matches or beats magnitude pruning WITHOUT post-hoc steps.
Experiment: MLP/conv; measure final sparsity-quality vs iterative magnitude pruning. L2.
Score: 6,5,6,6,5,5,6,7,5,4 → 5.5

### H46 — Adversarial Storekeeper
Generator produces candidates to remember; discriminator predicts future usefulness; game equilibrates
into good memory policy.
Math: min_G max_D E[log D(useful)] + E[log(1−D(G(z)))]; G proposes traces, D scores.
Prior: GANs (outputs), adversarial memory addressing (sparse access), prioritized replay (heuristic priority).
Difference: usefulness prediction trained ADVERSARIALLY against memory-proposal policy — co-evolving curator.
Prediction: converges to storing borderline cases (max information), avoiding redundant/easy traces.
Experiment: replay buffer curation; adversarial vs PER on sparse-feedback tasks. L2-3.
Score: 6,5,6,6,5,5,5,6,5,4 → 5.3

### H47 — Coevolving Store Specializations
Multiple stores coevolve complementary specializations via competitive pressure on shared prediction load.
Math: stores compete per-sample; winner takes gradient share; losers mutate structure slightly.
Prior: mixture density networks, boosting, MoE (all static-capacity).
Difference: structural mutation + competition creates differentiated STORES (not just gates).
Prediction: stores differentiate along interpretable axes (e.g., recency vs frequency vs modality).
Experiment: multi-statistics streams; track store specializations vs data axes. L2.
Score: 5,5,6,5,5,4,5,6,5,4 → 4.9

## FAMILY J — DEEP CONCEPTUAL REFRAMES

### H48 — Learning as Partial Evaluation
Training = partial evaluation (Futamura projections): interpreting data under current model compiles into
specialized fast paths; multiple projection levels = multiple timescales.
Math: proj_1(model,data)→specialized interpreter; proj_2(·)→further compiled; PL theory mapping.
Prior: Futamura (compilers); neural module networks (composition); compiled nets.
Difference: LEARNING TIMESCALES identified with projection LEVELS; promotion = further projection.
Prediction: projection hierarchy predicts which knowledge becomes fast vs stays interpretive.
Experiment: transformer w/ explicit interpretation layers; measure compilation gains per level. L3-4.
Score: 8,7,5,6,5,8,5,4,8,7 → 6.5

### H49 — Landauer-Priced Learning
Physical information erasure cost (kT ln2 per bit) as REAL constraint in objective; memory management
respects thermodynamic budget (simulated).
Math: add k_B T Σ erased_bits to loss; optimal forgetting = cheapest distortion.
Prior: thermodynamics of computation (theory); thermodynamic computing hardware proposals.
Difference: first ML training principle treating ERASURE COST as first-class economic term.
Prediction: induces bias toward compressible (structured) solutions — links physics to generalization.
Experiment: parity-vs-smooth function learning; check compressibility bias vs standard SGD. L3.
Score: 7,8,4,6,6,8,3,5,8,6 → 6.05

### H50 — Monoidal Learner Composition
Learners as objects in monoidal category; parallel/sequential composition laws define how learning
transfers; training = constructing functor.
Math: learners L1⊗L2 compose; update functors commute with composition; backprop = special case.
Prior: categorical DL (backprop as functor — Fong et al.), categorical foundation exists.
Difference: extends to LEARNED UPDATES and MEMORY composition, not just gradient structure.
Prediction: compositional transfer theorems yield concrete zero-shot assembly guarantees.
Experiment: verify predicted transfer bounds on composed modular nets. L4.
Score: 7,8,4,7,4,9,4,3,7,5 → 5.95

### H51 — Superposition Hypothesis Stores
Store maintains amplitude-weighted SUPERPOSITION of hypotheses; queries perform "measurement"
(collapsing to best-matching); interference = intended feature not bug.
Math: ψ=Σ_i a_i h_i; response=<ψ,Q>; amplitudes updated by Bayesian-like reinforcement.
Prior: quantum ML (mostly algorithms-on-qubits), superposition (representation-level), Hopfield mixtures.
Difference: deliberate quantum-LIKE algebra CLASSICALLY implemented for memory economies.
Prediction: interference patterns improve few-shot generalization vs clean-slot memories.
Experiment: correlated-task streams; superposed store vs slot store. L3.
Score: 5,6,4,4,5,6,3,5,6,5 → 4.75

### H52 — Memory Morphogenesis
Memory organization emerges via reaction-diffusion-like LOCAL dynamics (morphogen gradients over
parameter space marking zones for growth/pruning/promotion).
Math: ∂m/∂t = D∇²m + f(m, local_error); thresholds trigger local structural events.
Prior: self-organizing maps (topology-fixed), neural cellular automata (representations), developmental models.
Difference: STRUCTURAL memory events (grow/prune/route) emerge from local chemical-like signaling.
Prediction: stable spatial differentiation of memory zones emerges; robust to damage.
Experiment: 2D-grid memory layer; lesion tests; zone regeneration. L3.
Score: 6,6,4,4,5,5,4,5,6,5 → 4.85

---
## SCORING SUMMARY (weighted)
Top tier: H01 Memory Economics **8.35** · H16 Inference Ledger **7.55** · H13 Prequential **7.30**
Second: H32 RD-Allocation 7.00 · H34 Pred-Info 6.75 · H40 Changepoint-Demotion 6.75 · H18 Market 6.3 ·
H22 Alg-States 6.35 · H25 Self-U 6.35 · H09 Moment-Transfer 6.3 · H41 Cascade 6.3 · H48 Partial-Eval 6.5
Rest: 4.75–6.05.
