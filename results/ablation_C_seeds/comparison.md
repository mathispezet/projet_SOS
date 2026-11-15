# Run comparison

Runs: `results/lizard_regen_s0`, `results/lizard_regen_s1`, `results/lizard_regen_s2`, `results/lizard_regen_soblap_s0`, `results/lizard_regen_soblap_s1`, `results/lizard_regen_soblap_s2`, `results/lizard_regen_learned_s0`, `results/lizard_regen_learned_s1`, `results/lizard_regen_learned_s2`

## Growth and long-horizon stability

| condition | seeds | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 |
|---|---|---|---|---|---|
| regenerate | 3 | 0.000236 [0.000228, 0.000252] | 0.000192 [0.000167, 0.000229] | 4.01e-05 [2.62e-05, 5e-05] | 4e-05 [2.61e-05, 4.97e-05] |
| regenerate (sobel_lap) | 3 | 8.36e-05 [7.31e-05, 0.000103] | 3.91e-05 [3.38e-05, 4.91e-05] | 1.24e-05 [9.65e-06, 1.46e-05] | 1.23e-05 [9.65e-06, 1.46e-05] |
| regenerate (learned) | 3 | 0.000112 [0.000104, 0.000117] | 7.94e-05 [5.54e-05, 0.000103] | 2.62e-05 [1.7e-05, 3.33e-05] | 2.62e-05 [1.7e-05, 3.36e-05] |

## Damage recovery

Damage applied at step 100, then 200 steps; 20 trials per condition. The undamaged control is the mean MSE of intact growths at the same step and the residual ratio is the MSE at the end of the window divided by that control. A trial counts as recovered when its MSE is under 0.01 and within 2.0x of the control (floor 0.001) at the end of the window and stayed there; the recovery time is the first step of that sustained return. Each condition pools its training seeds (0, 1, 2): the cells give the mean over the seeds with the [min, max] range; a recovery rate is therefore averaged over seeds x trials.

| condition | damage | seeds | MSE at end of window | residual ratio | recovery rate | mean recovery time (steps) |
|---|---|---|---|---|---|---|
| regenerate | circle | 3 | 0.000413 [0.000354, 0.000451] | 10.6 [8.6, 13.3] | 0.88 [0.85, 0.90] | 39 [38, 41] |
| regenerate | half | 3 | 0.000272 [9.12e-05, 0.000383] | 6.2 [3.42, 8.66] | 0.90 [0.85, 1.00] | 67 [61, 71] |
| regenerate | dropout | 3 | 0.000118 [7.11e-05, 0.000186] | 2.93 [1.61, 3.63] | 0.98 [0.95, 1.00] | 24 [20, 26] |
| regenerate (sobel_lap) | circle | 3 | 7.04e-05 [1.05e-05, 0.000186] | 4.96 [1.08, 12.6] | 0.98 [0.95, 1.00] | 33 [32, 34] |
| regenerate (sobel_lap) | half | 3 | 0.000254 [1.04e-05, 0.000737] | 17.4 [1.07, 50.1] | 0.93 [0.80, 1.00] | 49 [49, 50] |
| regenerate (sobel_lap) | dropout | 3 | 1.29e-05 [1.01e-05, 1.55e-05] | 1.04 [1.01, 1.05] | 1.00 [1.00, 1.00] | 12 [10, 14] |
| regenerate (learned) | circle | 3 | 0.000298 [0.000188, 0.000488] | 11.7 [5.55, 16.5] | 0.92 [0.85, 0.95] | 34 [28, 36] |
| regenerate (learned) | half | 3 | 0.000227 [5.66e-05, 0.000463] | 8.36 [1.92, 13.7] | 0.97 [0.95, 1.00] | 56 [49, 70] |
| regenerate (learned) | dropout | 3 | 7.7e-05 [2.04e-05, 0.000176] | 2.53 [1.17, 5.21] | 0.98 [0.95, 1.00] | 17 [16, 18] |

## Figures

- `growth_curves.png`
- `damage_circle_curves.png`
- `damage_half_curves.png`
- `damage_dropout_curves.png`
- `growth_strips.png`
- `damage_circle_strips.png`
- `damage_half_strips.png`
- `damage_dropout_strips.png`
