# RESULTS NARRATIVE — What the Full Suite Actually Shows (honest version)

Written after the complete 10-seed run; supersedes optimistic quick-run impressions.

## Headline findings
1. **Value-based routing wins the aggregate.** Among capacity-limited methods (pes, clock, clock_mag,
   single_decay, random_routing — identical predictors/SGD/capacities), PES has the best mean rank
   across four nonstationarity regimes. The random-routing control ranks LAST ⇒ the value signal
   carries real information; sparsity alone does not explain results (kills competing hypothesis C-a).
2. **No per-regime dominance.** Only one comparison reaches significance in PES's favor
   (alternating vs clock_mag, d=+0.94, p=0.021). In stationary retention, random_routing significantly
   beats PES (d=-1.44, p=0.002). Honest conclusion: priced routing is a robust *aggregate* organizer,
   not a per-benchmark winner at this scale.
3. **The distinctive-signature program mostly failed as pre-registered — and that is informative:**
   - Bimodality (P2) is a property of having two stores, not of pricing (both methods BC≈0.73).
   - Spike-locking (P3) is ABSENT at default hysteresis — because switching costs suppress exactly
     the reactivity that would create it. This retrodiction was derived AFTER seeing the null and then
     CONFIRMED by an independent dose-response probe (see 4); it stands as theory+confirmation, not
     as a prediction, and we label it accordingly.
   - Reminiscence (P4) could not manifest by construction: cap_slow=15 < |support|=25 per rule set,
     so no trace survives between alternating visits. Design flaw, documented; test needs redesign.
   - Tariff law (P5): maintenance tariffs do NOT bind anywhere in the swept range (flat promotion rate);
     TRANSITION costs bind strongly (175× throughput change across hysteresis window; monotone κ response).
4. **The operative price of persistence is the cost of moving knowledge, not of keeping it.**
   This is the single cleanest empirical law of the study (dose-responsive, saturates where predicted).

## What this means for the paper claim
Surviving scientific core:
(a) unification table (degenerate routings) — conceptual contribution, unaffected;
(b) value-signal necessity under capacity constraints — supported (random-routing control);
(c) transition-cost-dominated routing dynamics — novel measurable law with dose-response;
(d) failure analyses of P2/P3/P4 — falsification-grade negative results that sharpen what any future
    "memory economics" claim must demonstrate (per-store pricing vs movement pricing are different
    mechanisms and must be tested separately).
Not survivable: claims of broad benchmark dominance or of distinctive emergent signatures at default
settings. These are removed from all drafts.

## Next steps queued (cron continuation)
R1: redesign P4 (cap_slow ≥ 2×|support| or sparser rules) — reminiscence remains theoretically sound.
R2': test P3 ONLY in the selective regime — first raise κ_move until promo rate falls well below the
   hysteresis budget (verify unsaturated), then measure lag-coupling. Probes R2/ρ established that
   saturated/frozen regimes mask any coupling (see FAILED_IDEAS F-09/F-10).
R3: formalize the THREE-REGIME routing law (saturated/selective/frozen) with the piecewise throughput
   model; this is now the strongest theory candidate from the experimental program.
R4-DONE: LR sweeps — drifting: PES best at lr∈{0.05,0.15} (beats clock & single_decay), loses only at
   lr=0.45; volatile: simpler methods win at high LR (protection tax under chaos). C-c partially
   closed: method ranking is LR-dependent but PES holds best-at-its-best parity in drift.
R5: torch port + MLP Phase-6 test once user approves install.

## Follow-up probe outcomes (all JSON-committed)
R2 hysteresis sweep: contrast 1.017/1.003/0.997 → no re-emergence (F-09).
ρ sweep: 0.996/1.000/0.994 at ρ∈{0.9,0.98,0.998}, promo counts identical → saturation discovered (F-10).
Dose-response: transition costs bind (175× throughput range); maintenance tariffs do not (P5 revised).
R2' selective-band test (r2prime.json): κ=0.5 saturated (rate .253>budget .2) mild contrast ~1.15;
κ=1.0 UNSATURATED (rate .055) mean contrast ≈1.17 but seed-spread 0.87/1.53/1.12 → SUGGESTIVE,
NOT SIGNIFICANT at n=3 (queue R2'' with n≥10 + pre-set threshold before any claim);
κ=2.0 frozen (rate .005) contrast ≈0.52 → frozen-band promotions are ANTI-coupled (tariff-floor
artifacts). Three-regime law now has per-regime signatures: which-trace correlates with errors when
saturated; when-to-move becomes value-driven when selective; timing decouples (inverts) when frozen.

## Verdict on P3 (final)
Not supported as originally registered. Current state: reactivity EXISTS only in the selective regime
and is weak/seed-variable there; claims require R2'' (n≥10, preregistered threshold, saturation check
per seed). Any surprise-gated-memory-style claim about THIS mechanism would be unsupported — and the
frozen-band anti-coupling actively contradicts a universal surprise-gating story.

## PHASE 6 — Neural-network port (torch_phase6.json, 2026-08-26)
Setup: 2-layer MLP (64 hidden) over stably-slot-bound resident inputs; same routing policies
(pes/random/clock/single); calibrated online-learning config (see RESEARCH_LOG D8); 3 regimes ×
4 policies × 3 seeds; T=8000; caps 55+15 on D=120/s0=10 task.

### What replicated
- **Value-signal necessity (core control result): YES, strongly.** Random-routing (identical machinery,
  noise valuation) is 3–5× worse than every valued policy in stationary (tails ≈0.11–0.12 vs 0.01–0.05)
  and ~2× worse under drift (≈0.30 vs 0.15–0.18). The economic *signal*, not the machinery, carries
  performance — now demonstrated on a nonlinear predictor too.
- **Protection tax under volatility: qualitatively consistent.** No method separates in the volatile
  regime (all tails 0.30–0.44); protection buys nothing under chaos, exactly as in the linear study.
- **[Expansion] Cross-family drift dominance at n=10:** under drift, PES significantly beats clock
  (9/10, p=0.021), single-decay (10/10, p=0.002), and noise-routing (9–10/10) in BOTH environment
  families; Gaussian features collapse unstructured decay (0.48 vs 0.16 stationary).
- **[R2''] Selective-band spike-locking: REJECTED at n=10** (median contrast 0.966, 4/10 over
  threshold, p=0.83). The n=3 "suggestive" signal was sampling noise. Migration timing is price-driven,
  not error-driven, across all regimes — a definitive negative that separates priced persistence from
  surprise gating.

### What did NOT replicate
- **PES aggregate dominance: NO at this scale.** Aggregate ranks: clock 2, pes 4, single 4, random 8.
  Clock wins stationary (0.014–0.028 vs PES 0.023–0.048); single-store decay edges drift. Differences
  among the three valued policies are small at n=3 (likely ns).
- Plausible mechanism (registered for R6): with caps(70) ≫ support(10), routing pressure is low and the
  input-gradient valuation through an MLP is noisier than exact linear responsibility — compressing
  policy differences. Testable prediction: shrink caps toward support size and PES-vs-clock gap should
  re-open (capacity-pressure dependence of policy advantage).

### Net scientific reading
The PRINCIPLE (value-priced persistence beats unpriced persistence) transfers to nonlinear learners;
the specific greedy-exchange POLICY does not currently beat simpler fixed schedules there. For the
paper: claim scope narrows to "value economics necessary, policy design open" — which is itself a
clean, falsifiable position. Queue R6 (capacity-pressure sweep on MLP) before any stronger statement.

## PHASE 6 / R6 FINAL (2026-08-26)
- Phase 6 (MLP): value-signal necessity replicates (random 3–5× worse); policy dominance does not.
- R6 capacity-pressure sweep: REGISTERED PREDICTION FALSIFIED — tighter capacity amplifies clock's
  advantage instead of re-opening PES's (14+6 stationary: 0.0127 vs 0.1004). Mechanism: pressured
  routing multiplies first-order credit-assignment errors through the nonlinearity (F-11).
- Standing scope: **value economics necessary; naive first-order pricing insufficient on nonlinear
  substrates.** Central open problem: trustworthy valuations under nonlinearity (influence estimates,
  leave-one-out probes, learned critics, longer integration windows).
- Queued next: R7 (valuation upgrade: replace ∂ŷ/∂x·x with cheap leave-one-out re-score at routing
  moments only), R2'' (selective-band significance n≥10), R1 (P4 redesign).

## Reproduction
research/code/README.md · seeds stable via crc32 · unit tests gate changes · JSON artifacts committed.
Torch port: research/code/pes_torch.py (calibration chain in results/mlp_diag*.txt).
