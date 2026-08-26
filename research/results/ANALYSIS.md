# EXPERIMENTAL ANALYSIS (auto-generated)

Config: T=3000, D=200, s0=25, eta=0.15, caps=[45, 15], seeds=10

## Main comparison (loss tail)

| method | stationary | drifting | volatile | alternating |
|---|---|---|---|---|
| sgd_dense | 0.0031±0.0022 | 0.2147±0.1151 | 0.4139±0.0924 | 0.0030±0.0020 |
| sgd_l2 | 0.0202±0.0041 | 0.1636±0.0827 | 0.3164±0.0828 | 0.0212±0.0058 |
| pes | 0.1606±0.0568 | 0.2672±0.1012 | 0.2682±0.0771 | 0.1796±0.0648 |
| random_routing | 0.2496±0.1017 | 0.2836±0.0989 | 0.2844±0.0796 | 0.2260±0.0804 |
| clock | 0.1722±0.0865 | 0.2814±0.1140 | 0.2842±0.0751 | 0.1590±0.0531 |
| clock_mag | 0.1672±0.0839 | 0.2831±0.1138 | 0.2847±0.0755 | 0.1579±0.0533 |
| single_decay | 0.2627±0.0894 | 0.2787±0.1085 | 0.2755±0.0761 | 0.2280±0.0855 |

## Rare recall MSE

| method | stationary | drifting | volatile | alternating |
|---|---|---|---|---|
| sgd_dense | 0.473 | 1.173 | 1.975 | 0.341 |
| sgd_l2 | 0.641 | 1.154 | 1.769 | 0.470 |
| pes | 2.027 | 2.368 | 2.365 | 2.062 |
| random_routing | 2.271 | 2.754 | 2.609 | 2.395 |
| clock | 1.666 | 2.803 | 2.412 | 1.392 |
| clock_mag | 1.615 | 2.811 | 2.409 | 1.435 |
| single_decay | 2.707 | 2.899 | 2.461 | 2.386 |

## Pre-registered predictions

**P1** (no fixed profile dominates PES everywhere; PES best on aggregate rank):
- aggregate ranks (lower=better): pes:2, clock:7, clock_mag:8, single_decay:10, random_routing:13
- stationary: random_routing: d=-1.44, p=0.002; clock: d=-0.15, p=0.623; clock_mag: d=-0.09, p=0.754; single_decay: d=-1.86, p=0.002
- drifting: random_routing: d=-0.52, p=0.109; clock: d=-0.50, p=0.754; clock_mag: d=-0.56, p=0.754; single_decay: d=-0.36, p=0.754
- volatile: random_routing: d=-0.89, p=0.344; clock: d=-0.81, p=0.109; clock_mag: d=-0.91, p=0.109; single_decay: d=-0.55, p=0.754
- alternating: random_routing: d=-1.43, p=0.002; clock: d=+0.83, p=0.109; clock_mag: d=+0.94, p=0.021; single_decay: d=-1.71, p=0.002

**P2** bimodality: pes BC=0.731±0.083 (n=40), clock BC=0.729±0.092 (n=40); threshold 0.555.

**P3** spike-locking: see fig4; PES curve should exceed 1.0 at short lags if promotions follow large errors. Values at lag 5: pes:1.01, clock:1.12, random_routing:1.00.

**P4** reminiscence: compare reacquire time block1 vs later blocks (alternating regime):
- first-block mean=1.4 steps, later-block mean=2.0; difference d=+0.22, sign p=0.109 (negative diff = faster re-acquisition)

**P5** tariff law: promotion rate vs tariffs — see fig6; monotone decrease with both tariffs expected.