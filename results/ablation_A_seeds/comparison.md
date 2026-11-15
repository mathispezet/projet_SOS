# Run comparison

Runs: `results/lizard_grow_s0`, `results/lizard_grow_s1`, `results/lizard_grow_s2`, `results/lizard_persist_s0`, `results/lizard_persist_s1`, `results/lizard_persist_s2`, `results/lizard_regen_s0`, `results/lizard_regen_s1`, `results/lizard_regen_s2`

## Growth and long-horizon stability

| condition | seeds | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 |
|---|---|---|---|---|---|
| grow | 3 | 0.000168 [0.000137, 0.000208] | 0.000119 [8.49e-05, 0.000166] | 0.0278 [0.0218, 0.0333] | 29.4 [0.068, 88.1] |
| persist | 3 | 4.19e-05 [3.57e-05, 4.99e-05] | 8.09e-05 [6.05e-05, 0.000103] | 1.07e-05 [7.63e-06, 1.33e-05] | 1.08e-05 [7.64e-06, 1.33e-05] |
| regenerate | 3 | 0.000236 [0.000228, 0.000252] | 0.000192 [0.000167, 0.000229] | 4.01e-05 [2.62e-05, 5e-05] | 4e-05 [2.61e-05, 4.97e-05] |

## Damage recovery

Damage applied at step 100, then 200 steps; 20 trials per condition. The undamaged control is the mean MSE of intact growths at the same step and the residual ratio is the MSE at the end of the window divided by that control. A trial counts as recovered when its MSE is under 0.01 and within 2.0x of the control (floor 0.001) at the end of the window and stayed there; the recovery time is the first step of that sustained return. Each condition pools its training seeds (0, 1, 2): the cells give the mean over the seeds with the [min, max] range; a recovery rate is therefore averaged over seeds x trials.

| condition | damage | seeds | MSE at end of window | residual ratio | recovery rate | mean recovery time (steps) |
|---|---|---|---|---|---|---|
| grow | circle | 3 | 0.0128 [0.0118, 0.0143] | 1.13 [1.07, 1.24] | 0.48 [0.25, 0.65] | 32 [26, 38] |
| grow | half | 3 | 0.00991 [0.0093, 0.0106] | 0.88 [0.788, 0.954] | 0.58 [0.50, 0.65] | 54 [52, 59] |
| grow | dropout | 3 | 0.00985 [0.0081, 0.0108] | 0.869 [0.803, 0.973] | 0.60 [0.40, 0.85] | 39 [39, 41] |
| persist | circle | 3 | 0.00833 [0.00612, 0.0104] | 770 [633, 875] | 0.57 [0.40, 0.70] | 42 [36, 47] |
| persist | half | 3 | 0.00751 [0.00383, 0.0124] | 672 [466, 1.05e+03] | 0.22 [0.00, 0.40] | 154 [148, 159] |
| persist | dropout | 3 | 0.00754 [0.0045, 0.00958] | 674 [591, 720] | 0.08 [0.00, 0.25] | 97 |
| regenerate | circle | 3 | 0.000413 [0.000354, 0.000451] | 10.6 [8.6, 13.3] | 0.88 [0.85, 0.90] | 39 [38, 41] |
| regenerate | half | 3 | 0.000272 [9.12e-05, 0.000383] | 6.2 [3.42, 8.66] | 0.90 [0.85, 1.00] | 67 [61, 71] |
| regenerate | dropout | 3 | 0.000118 [7.11e-05, 0.000186] | 2.93 [1.61, 3.63] | 0.98 [0.95, 1.00] | 24 [20, 26] |

## Figures

- `growth_curves.png`
- `damage_circle_curves.png`
- `damage_half_curves.png`
- `damage_dropout_curves.png`
- `growth_strips.png`
- `damage_circle_strips.png`
- `damage_half_strips.png`
- `damage_dropout_strips.png`
