# PRE-SUBMISSION PRIOR-ART SWEEP #5 (2026-08-26) + VERDICT

## New neighbors found (not seen in waves 1–4)

### 2606.12945 — "Learning What to Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory" (Chen & Cheng)
**Title-level overlap is real; mechanism overlap is not.**
- Their object: LLM-agent *textual* memories; a 7-factor interpretable value function
  (emotion, goal relevance, self-relevance, task utility, reliability, usage...) with weights learned by
  gradient-free optimization on LongMemEval retention. Cognitive-psychology grounded.
- Shared with us: the normative question "what deserves to be kept", learned per-item valuation,
  blind-at-consolidation framing, single scalar controlling encode/forget/retrieve.
- Different: substrate (text events vs gradient traces), no store multiplicity, no maintenance tariffs,
  no switching costs, no movement between media at all, no dynamical signatures. Their "value" is
  behavioral/cognitive factors; ours is predictive responsibility under priced persistence.
- **Action: MUST CITE as nearest concurrent work; must add to Related Work §2 and to the degenerate-
  routings discussion (their system = single-store routing where only content value varies).**
- Novelty impact: does not kill the claim (no cross-store economics anywhere in it), but any reviewer who
  knows this paper will demand the comparison. Having it cited preempts that.

### 2604.18002 — "Neural Garbage Collection" (Li) [RESOLVED — read in full]
RL-learned KV-cache eviction for chain-of-thought reasoning, trained from outcome reward alone.
Single store (the KV cache), eviction-only decisions, no store multiplicity, no maintenance tariffs,
no switching costs. Family: one-store learned eviction alongside OBCache and DeltaNet-style gating.
**Action: cite as another degenerate single-store pricing; not a novelty threat. Sweep flag cleared.**

### Other sweep hits (lower threat)
- GradMem (2603.13875): test-time gradient writes into context memory — write-gating family, covered.
- OBCache / Cache-What-Lastes / tiered KV storage (2510/2512/2603): KV eviction heuristics with
  importance scores — single-store, systems-flavored; citable as engineering instances of one-store
  pricing. Worth citing in a camera-ready related-work expansion.
- RL-for-memory-management line (2410.15492, CMI-Mem, Mem-α): agents managing textual memory with RL;
  different substrate and objective; optional cite.

## Standing verdict after 5 sweeps (~150 queries total)
The specific claim — ONE objective pricing keep-and-move across persistence-heterogeneous stores of a
gradient learner, with derived special cases and regime laws — remains **apparently novel**, now with a
documented nearest-neighbor set including two concurrent works that share fragments (lifecycle intuition
at systems level: 2608.22215; learned item-value for single-store text memory: 2606.12945).
Language stays at "apparently novel"; the paper's §2 must absorb both citations before submission.
