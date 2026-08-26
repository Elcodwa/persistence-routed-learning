# FIELD MAP — Literature Investigation (Waves 1–3, 2026-08-25)

~95 programmatic queries (web_search) + targeted arXiv extractions. Sources = URLs below; claims kept
to what titles/abstracts/fetched text support. Status labels: [verified-read] = fetched full/abstract text;
[snippet] = title/snippet only.

## A. Sequence modeling & memory mechanisms
- Linear transformers ≡ fast weight programmers (outer-product associative memory). [snippet]
  https://www.emergentmind.com/topics/linear-transformers-as-fast-weight-programmers ;
  Irie & Gershman 2026 review: https://gershmanlab.com/pubs/IrieGershman26.pdf ; Schmidhuber 1991 roots:
  https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer-bigrefs.html
- Mamba selective SSMs: content-based selection weakness identified & patched via input-dependent Δ.
  [snippet] https://arxiv.org/pdf/2312.00752
- Titans: Learning to Memorize at Test Time — neural long-term memory, surprise-based updates.
  [snippet] https://openreview.net/forum?id=8GjSf9Rh7Z
- TTT: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" — hidden state = learned
  model updated by test-time gradient steps. [snippet] https://arxiv.org/abs/2407.04620 ;
  NVIDIA: "TTT with KV Binding Is Secretly Linear Attention" https://research.nvidia.com/labs/sil/projects/tttla/
- Gated Delta Networks: Mamba2 + delta rule, input-dependent gating. [snippet]
  https://arxiv.org/pdf/2412.06464
- Modern Hopfield ≡ attention; linear attention ≡ continually-updated Hopfield. [snippet]
  https://arxiv.org/abs/2008.02217 ; https://www.beren.io/2024-03-03-Linear-Attention-as-Iterated-Hopfield-Networks/
- Dense Associative Memories exponential capacity. [snippet]
  https://link.aps.org/doi/10.1103/PhysRevLett.132.077301
- KV cache compression surveys (engineering maturity of "context as memory"). [snippet]
  https://arxiv.org/html/2508.06297v1
- Long-context vs RAG: lost-in-the-middle, recency bias, effective-context limits. [snippet]
  https://arxiv.org/html/2407.16833v1
- kNN-LM limits: "Great Memory, Shallow Reasoning". [snippet] https://aclanthology.org/2025.naacl-short.40.pdf

## B. Adaptation, continual learning, plasticity
- Test-time adaptation surveys (entropy minimization et al.). [snippet] https://arxiv.org/abs/2303.15361
- Plasticity loss: Nature 2024 "Loss of plasticity in deep continual learning" (continual backprop).
  [snippet] https://www.nature.com/articles/s41586-024-07711-7 ; primacy bias; Zyphra analysis.
- Synaptic Intelligence (per-parameter importance for protection). [verified-abstract]
  https://arxiv.org/abs/1703.04200
- Complex synapses (Kaplanis et al.) — multi-timescale synaptic variables for continual RL. [snippet]
  https://proceedings.mlr.press/v80/kaplanis18a.html
- Fast/slow synaptic plasticity enables concurrent control (eLife reviewed preprint). [snippet]
  https://elifesciences.org/reviewed-preprints/105043
- Fast & Slow Variational Continual Learning. [snippet] https://arxiv.org/pdf/2606.24007
- Hypernetworks for continual learning (task-conditioned weight generation). [snippet]
  https://arxiv.org/abs/1906.00695
- Model editing surveys (ROME/MEMIT lineage). [snippet] https://arxiv.org/abs/2310.16218

## C. Optimization as learning/memory
- Nested Learning: models = nested optimization problems w/ own context flows; optimizers = associative
  memory compressing gradients; Hope; self-modifying module. NeurIPS 2025. [verified-abstract]
  https://arxiv.org/abs/2512.24695 ; blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- Learned optimizers: Veeriah et al.; reverse-engineering learned optimizers; practical tradeoffs;
  CeLO. [snippet] https://arxiv.org/abs/2011.02159 ; https://arxiv.org/abs/2203.11860
- Lookahead (slow/fast weight pairs). [snippet] https://www.researchgate.net/publication/334602765
- Muon momentum as spectral filtering / denoising. [snippet] https://arxiv.org/abs/2606.03899
- Two-timescale stochastic approximation (Borkar lineage; 2024 general conditions). [snippet]
  https://arxiv.org/abs/2412.19872
- Adam instability theory at scale (time-domain Hessian correlations → divergence/loss spikes). [verified-read]
  https://arxiv.org/pdf/2304.09871
- Convergence of optimizers ⇒ eigenvalue filtering at equilibrium. [snippet] https://arxiv.org/html/2510.09034

## D. Train/inference boundary
- ICL as implicit Bayesian inference. [snippet] https://arxiv.org/abs/2111.02080
- "Transformers learn in-context by gradient descent" + rebuttal "ICL and Gradient Descent Revisited".
  [snippet] https://arxiv.org/abs/2212.07677 ; https://arxiv.org/abs/2311.07772
- Induction-head formation: three interacting subcircuits cause phase change. [verified-read]
  https://arxiv.org/pdf/2404.07129
- Test-time scaling surveys (compute at inference ≠ weight change). [snippet]
  https://arxiv.org/abs/2503.24235
- TTT on nearest neighbors for LLMs. [snippet] https://arxiv.org/pdf/2305.18466

## E. Consolidation / memory systems (ML + neuro)
- Complementary Learning Systems: 3-stage consolidation model; mathematical CLS theory (Nature Neuro 2023
  "Organizing memories for generalization"). [snippet] https://www.nature.com/articles/s41593-023-01382-9
- Why the Brain Consolidates: Predictive Forgetting for Optimal Generalisation — consolidation = selective
  reduction of I(X;Z|Y) preserving I(Y;Z); keys stable / values compressed; replay necessity from readout
  complexity. [verified-read] https://arxiv.org/pdf/2603.04688
- GENESIS: generative episodic–semantic interaction model. [snippet] https://arxiv.org/pdf/2510.15828v2
- Hippocampus as hierarchical generative model supporting generative replay. [snippet]
  https://www.sciencedirect.com/science/article/abs/pii/S0301008222001150
- Hinton & Plaut 1987: fast weights for recent past. [snippet]
  https://cnbc.cmu.edu/~plaut/papers/abstracts/HintonPlaut87CogSciConf.fastWeights.html ;
  Ba et al. 2016 fast weights. https://proceedings.neurips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past.pdf

## F. Foundations / dynamics / phenomena
- Information bottleneck debates (fitting vs compression phases contested). [snippet]
  https://arxiv.org/abs/1503.02406
- Grokking: weight-norm causal delay law; Fourier circuits; memorization↔generalization transitions.
  [snippet] https://arxiv.org/html/2606.13753v1 ; https://arxiv.org/abs/2301.02679
- Superposition & polysemanticity (capacity pressures). [snippet] https://arxiv.org/abs/2210.01892
- Double descent. [snippet] https://arxiv.org/abs/1912.02292
- Data-constrained scaling laws (repeated epochs, diminishing returns). [snippet]
  https://arxiv.org/abs/2305.16264
- Lazy/NTK vs rich feature-learning regimes. [snippet] https://arxiv.org/pdf/2404.19719
- Predictive forgetting (see E) — normative forgetting.
- Volatility-adaptive learning: "A simple model for learning in volatile environments" (uncertainty-modulated
  gain). [snippet] https://ncbi.nlm.nih.gov/pmc/articles/PMC7329063
- Equilibrium propagation (+ dissipative extensions). [snippet]
  https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full
- Forward-forward status: competitive on small conv but unproven at scale. [snippet]
  https://arxiv.org/pdf/2511.01061v1
- Meta-plasticity (synaptic metaplasticity in binarized nets; meta-learned plasticity). [snippet]
  https://www.nature.com/articles/s41467-021-22768-y ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8813911/
- Self-referential weight matrix that learns to modify itself (Irie et al. ICML 2022). [snippet]
  https://proceedings.mlr.press/v162/irie22b/irie22b.pdf
- Gödel machine (provably optimal self-improvement; intractable in practice). [snippet]
  https://people.idsia.ch/~juergen/goedelmachine.html
- Weight agnostic networks. [snippet] https://arxiv.org/pdf/1906.04358
