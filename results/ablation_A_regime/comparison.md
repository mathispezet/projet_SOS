# Run comparison

Runs: `results/lizard_grow_s0`, `results/lizard_persist_s0`, `results/lizard_regen_s0`

## Growth and long-horizon stability

| run | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 | training (min) |
|---|---|---|---|---|---|
| grow | 0.000137 | 8.49e-05 | 0.0333 | 88.1 | 20.2 |
| persist | 3.57e-05 | 6.05e-05 | 7.63e-06 | 7.64e-06 | 15.4 |
| regenerate | 0.000252 | 0.000229 | 5e-05 | 4.97e-05 | 15.7 |

## Damage recovery

Damage applied at step 100, then 200 steps; 20 trials per condition. The undamaged control is the mean MSE of intact growths at the same step and the residual ratio is the MSE at the end of the window divided by that control. A trial counts as recovered when its MSE is under 0.01 and within 2.0x of the control (floor 0.001) at the end of the window and stayed there; the recovery time is the first step of that sustained return.

| run | damage | MSE after damage | MSE at end of window | undamaged control | residual ratio | recovery rate | mean recovery time (steps) |
|---|---|---|---|---|---|---|---|
| grow | circle | 0.00872 | 0.0143 | 0.0134 | 1.07 | 0.25 | 38 |
| grow | half | 0.0153 | 0.0106 | 0.0134 | 0.788 | 0.50 | 59 |
| grow | dropout | 0.0073 | 0.0108 | 0.0134 | 0.803 | 0.40 | 41 |
| persist | circle | 0.00868 | 0.00612 | 7.62e-06 | 802 | 0.70 | 41 |
| persist | half | 0.0151 | 0.00383 | 7.62e-06 | 502 | 0.25 | 159 |
| persist | dropout | 0.00734 | 0.0045 | 7.62e-06 | 591 | 0.25 | 97 |
| regenerate | circle | 0.00884 | 0.000451 | 5.24e-05 | 8.6 | 0.90 | 41 |
| regenerate | half | 0.0153 | 0.000342 | 5.24e-05 | 6.52 | 0.85 | 68 |
| regenerate | dropout | 0.0074 | 0.000186 | 5.24e-05 | 3.55 | 1.00 | 26 |

## Figures

- `growth_curves.png`
- `damage_circle_curves.png`
- `damage_half_curves.png`
- `damage_dropout_curves.png`
- `growth_strips.png`
- `damage_circle_strips.png`
- `damage_half_strips.png`
- `damage_dropout_strips.png`
