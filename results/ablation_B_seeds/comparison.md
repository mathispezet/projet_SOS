# Run comparison

Runs: `results/lizard_regen_p100_s0`, `results/lizard_regen_p100_s1`, `results/lizard_regen_p100_s2`, `results/lizard_regen_s0`, `results/lizard_regen_s1`, `results/lizard_regen_s2`, `results/lizard_regen_p025_s0`, `results/lizard_regen_p025_s1`, `results/lizard_regen_p025_s2`

## Growth and long-horizon stability

| condition | seeds | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 |
|---|---|---|---|---|---|
| p = 1.0 | 3 | 8.29e-05 [6.38e-05, 9.96e-05] | 2.95e-05 [2.09e-05, 3.6e-05] | 2e-05 [1.42e-05, 2.47e-05] | 2e-05 [1.42e-05, 2.47e-05] |
| p = 0.5 | 3 | 0.000236 [0.000228, 0.000252] | 0.000192 [0.000167, 0.000229] | 4.01e-05 [2.62e-05, 5e-05] | 4e-05 [2.61e-05, 4.97e-05] |
| p = 0.25 | 3 | 0.000821 [0.00071, 0.000963] | 0.00125 [0.00114, 0.00142] | 0.000142 [0.000101, 0.000216] | 0.000148 [8.59e-05, 0.000256] |

## Damage recovery

Damage applied at step 100, then 200 steps; 20 trials per condition. The undamaged control is the mean MSE of intact growths at the same step and the residual ratio is the MSE at the end of the window divided by that control. A trial counts as recovered when its MSE is under 0.01 and within 2.0x of the control (floor 0.001) at the end of the window and stayed there; the recovery time is the first step of that sustained return. Each condition pools its training seeds (0, 1, 2): the cells give the mean over the seeds with the [min, max] range; a recovery rate is therefore averaged over seeds x trials.

| condition | damage | seeds | MSE at end of window | residual ratio | recovery rate | mean recovery time (steps) |
|---|---|---|---|---|---|---|
| p = 1.0 | circle | 3 | 3.78e-05 [2.47e-05, 4.74e-05] | 2.06 [1.18, 3.32] | 1.00 [1.00, 1.00] | 32 [32, 32] |
| p = 1.0 | half | 3 | 0.00161 [9.52e-05, 0.00291] | 70.6 [6.67, 118] | 0.70 [0.55, 1.00] | 57 [49, 63] |
| p = 1.0 | dropout | 3 | 3.45e-05 [1.62e-05, 5.61e-05] | 1.69 [1.13, 2.68] | 1.00 [1.00, 1.00] | 24 [21, 30] |
| p = 0.5 | circle | 3 | 0.000413 [0.000354, 0.000451] | 10.6 [8.6, 13.3] | 0.88 [0.85, 0.90] | 39 [38, 41] |
| p = 0.5 | half | 3 | 0.000272 [9.12e-05, 0.000383] | 6.2 [3.42, 8.66] | 0.90 [0.85, 1.00] | 67 [61, 71] |
| p = 0.5 | dropout | 3 | 0.000118 [7.11e-05, 0.000186] | 2.93 [1.61, 3.63] | 0.98 [0.95, 1.00] | 24 [20, 26] |
| p = 0.25 | circle | 3 | 0.00139 [0.000857, 0.00166] | 6.87 [5.03, 9.36] | 0.70 [0.60, 0.75] | 52 [48, 56] |
| p = 0.25 | half | 3 | 0.000888 [0.000666, 0.00128] | 4.26 [3.86, 4.85] | 0.68 [0.40, 0.90] | 119 [113, 124] |
| p = 0.25 | dropout | 3 | 0.000457 [0.000256, 0.000712] | 2.1 [1.86, 2.27] | 0.92 [0.80, 1.00] | 57 [47, 61] |

## Figures

- `growth_curves.png`
- `damage_circle_curves.png`
- `damage_half_curves.png`
- `damage_dropout_curves.png`
- `growth_strips.png`
- `damage_circle_strips.png`
- `damage_half_strips.png`
- `damage_dropout_strips.png`
