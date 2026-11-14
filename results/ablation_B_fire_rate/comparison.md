# Run comparison

Runs: `results/lizard_regen_p100_s0`, `results/lizard_regen_s0`, `results/lizard_regen_p025_s0`

## Growth and long-horizon stability

| run | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 | training (min) |
|---|---|---|---|---|---|
| p = 1.0 | 8.54e-05 | 3.17e-05 | 2.11e-05 | 2.12e-05 | 15.1 |
| p = 0.5 | 0.000252 | 0.000229 | 5e-05 | 4.97e-05 | 15.7 |
| p = 0.25 | 0.00079 | 0.00119 | 0.000101 | 8.59e-05 | 15.0 |

## Damage recovery

Damage applied at step 100, then 200 steps; 20 trials per condition. The undamaged control is the mean MSE of intact growths at the same step and the residual ratio is the MSE at the end of the window divided by that control. A trial counts as recovered when its MSE is under 0.01 and within 2.0x of the control (floor 0.001) at the end of the window and stayed there; the recovery time is the first step of that sustained return.

| run | damage | MSE after damage | MSE at end of window | undamaged control | residual ratio | recovery rate | mean recovery time (steps) |
|---|---|---|---|---|---|---|---|
| p = 1.0 | circle | 0.00847 | 2.47e-05 | 2.09e-05 | 1.18 | 1.00 | 32 |
| p = 1.0 | half | 0.0147 | 0.00181 | 2.09e-05 | 86.7 | 0.55 | 49 |
| p = 1.0 | dropout | 0.00646 | 5.61e-05 | 2.09e-05 | 2.68 | 1.00 | 30 |
| p = 0.5 | circle | 0.00884 | 0.000451 | 5.24e-05 | 8.6 | 0.90 | 41 |
| p = 0.5 | half | 0.0153 | 0.000342 | 5.24e-05 | 6.52 | 0.85 | 68 |
| p = 0.5 | dropout | 0.0074 | 0.000186 | 5.24e-05 | 3.55 | 1.00 | 26 |
| p = 0.25 | circle | 0.00982 | 0.000857 | 0.000138 | 6.23 | 0.75 | 56 |
| p = 0.25 | half | 0.016 | 0.000666 | 0.000138 | 4.85 | 0.90 | 124 |
| p = 0.25 | dropout | 0.00905 | 0.000256 | 0.000138 | 1.86 | 1.00 | 61 |

## Figures

- `growth_curves.png`
- `damage_circle_curves.png`
- `damage_half_curves.png`
- `damage_dropout_curves.png`
- `growth_strips.png`
- `damage_circle_strips.png`
- `damage_half_strips.png`
- `damage_dropout_strips.png`
