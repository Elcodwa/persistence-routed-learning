# Propositions for PES Routing (with proofs)

These upgrade the informal "greedy exchange is optimal" claim into precise statements with proofs,
plus explicit boundaries where optimality fails. Added to paper §3.3.

---

## Setup

Traces $i \in \{1..K\}$, stores $s \in \{f, s\}$ (fast/slow) with integer capacities $M_f, M_s$,
$M_f + M_s \ge K' $ for the number of resident traces considered. Each trace carries a scalar value
$v_i \ge 0$. Placement scores are **additively separable**:

$$\Pi(i,s) = v_i - \lambda_s - \mu_s \,\mathrm{staleness}_s(i)$$

so that the total score of a placement $\sigma$ (a partial assignment of traces to stores) is
$\Sigma(\sigma) = \sum_i \Pi(i, \sigma(i))$ over assigned traces. A placement is *feasible* if
$|\sigma^{-1}(s)| \le M_s$ for each store.

## Proposition 1 (Static optimality of top-$M$ routing).

*If every resident trace must be placed and utilities are additive-separable as above, then the
placement that puts, in each store $s$, the $M_s$ highest-scoring traces by $\Pi(\cdot, s)$ subject to
joint feasibility maximizes $\Sigma$ over feasible placements. Moreover, any feasible placement that is
not of this form admits a single swap that strictly increases $\Sigma$.*

**Proof.** Since $\Sigma$ decomposes per trace and capacities couple traces only through counts, the
problem is an assignment problem with identical-cost slots inside each store: for fixed occupancy
numbers $(k_f, k_s)$, the optimum assigns to each store the $k_s$ traces with largest $\Pi(\cdot, s)$
(any other choice swaps two traces within a store and loses score). Over all occupancy vectors with
$k_f + k_s = n$ (all residents placed), $\Sigma$ is maximized by choosing the split that the score
differences dictate; exchanging one occupant of $f$ with one occupant of $s$ changes $\Sigma$ by
$[\Pi(i, s) - \Pi(i, f)] - [\Pi(j, f) - \Pi(j, s)]$, and any configuration not sorted by these margins
is improved by such an exchange. Because the exchange graph over feasible placements is connected and
every non-maximal vertex has an improving neighbor, hill-climbing by single exchanges reaches the
maximum; equivalently the constraint matrix is the incidence matrix of a transversal matroid pair
(interchange structure), whose bases satisfy the exchange axiom used above. $\square$

**Corollary (eviction rule).** Under capacity pressure in store $s$, the trace with minimum
$\Pi(\cdot, s)$ is the unique correct eviction target.

## Proposition 2 (Dynamic myopia with switching fees).

*Let the router act only at window openings, may perform at most one fast↔slow exchange per opening,
and pays fee $\kappa$ against the moved trace's value for each migration. If values $v_i(t)$ follow any
arbitrary sequence (adversarial streaming model), the greedy exchange rule — swap iff the post-fee gain*
$\max_i [\Pi_t(i, s)] - \min_j [\Pi_t(j, f)] > 2\kappa$ *— is exactly optimal among policies restricted
to one movement per opening, against the realized sequence.*

**Proof.** At an opening, any one-move policy either moves nothing (payoff 0 change) or moves some pair
$(i \to s, j \to f)$, changing the running score by
$\Delta(i,j) = \Pi_t(i,s) + \Pi_t(j,f) - \Pi_t(i,f) - \Pi_t(j,s)$ minus fees $2\kappa$ charged against
future scoring through reduced $v$. Because the sequence after the opening is arbitrary, no policy can
condition profitably on it; the best one-move action is therefore the myopic maximizer of realized
$\Delta - 2\kappa$, which is precisely the rule stated (the max–min form follows from separability:
given that a swap occurs, the best donor is $\arg\max_i \Pi_t(i,s)$ and the best evictee
$\arg\min_j \Pi_t(j,f)$). Moving nothing dominates any negative-gain swap. $\square$

## Where optimality breaks (stated plainly)

1. **Synergy.** If $\Pi$ is not additive (trace usefulness depends on which other traces are resident),
   Proposition 1 fails; maximum-coverage-style submodularity/supermodularity issues arise, and greedy is
   approximate at best. Our environments have additive ground truth; synergistic settings are untested.
2. **Multi-step look-ahead.** Proposition 2's optimality class excludes policies that skip a beneficial
   swap now to enable a better one later. With predictable value dynamics, saving migration budget can
   dominate; our streaming model assumes unpredictability, matching online-learning convention but not,
   e.g., slowly-varying deterministic drift with known phase.
3. **Price quality.** Both propositions condition on the scores being correct. Section 5.5/R6 shows the
   entire advantage structure collapses when $\Pi$ misattributes credit through a nonlinearity — the
   theorems are about allocation given prices, not about price discovery.
