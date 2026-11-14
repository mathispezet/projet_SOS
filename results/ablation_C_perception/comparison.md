# Run comparison

Runs: `results/lizard_regen_s0`, `results/lizard_regen_soblap_s0`, `results/lizard_regen_learned_s0`

## Growth and long-horizon stability

| run | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 | training (min) |
|---|---|---|---|---|---|
| sobel | 0.000252 | 0.000229 | 5e-05 | 4.97e-05 | 15.7 |
| sobel + laplacian | 0.000103 | 4.91e-05 | 1.46e-05 | 1.46e-05 | 15.3 |
| learned | 0.000117 | 0.000103 | 3.33e-05 | 3.36e-05 | 15.4 |

## Damage recovery

Damage applied at step 100, then 200 steps; 20 trials per condition. The undamaged control is the mean MSE of intact growths at the same step and the residual ratio is the MSE at the end of the window divided by that control. A trial counts as recovered when its MSE is under 0.01 and within 2.0x of the control (floor 0.001) at the end of the window and stayed there; the recovery time is the first step of that sustained return.

| run | damage | MSE after damage | MSE at end of window | undamaged control | residual ratio | recovery rate | mean recovery time (steps) |
|---|---|---|---|---|---|---|---|
| sobel | circle | 0.00884 | 0.000451 | 5.24e-05 | 8.6 | 0.90 | 41 |
| sobel | half | 0.0153 | 0.000342 | 5.24e-05 | 6.52 | 0.85 | 68 |
| sobel | dropout | 0.0074 | 0.000186 | 5.24e-05 | 3.55 | 1.00 | 26 |
| sobel + laplacian | circle | 0.0086 | 0.000186 | 1.47e-05 | 12.6 | 0.95 | 32 |
| sobel + laplacian | half | 0.015 | 0.000737 | 1.47e-05 | 50.1 | 0.80 | 49 |
| sobel + laplacian | dropout | 0.00682 | 1.55e-05 | 1.47e-05 | 1.05 | 1.00 | 14 |
| learned | circle | 0.00871 | 0.000188 | 3.38e-05 | 5.55 | 0.95 | 36 |
| learned | half | 0.0151 | 0.000463 | 3.38e-05 | 13.7 | 0.95 | 70 |
| learned | dropout | 0.0069 | 0.000176 | 3.38e-05 | 5.21 | 0.95 | 18 |

## Figures

- `growth_curves.png`
- `damage_circle_curves.png`
- `damage_half_curves.png`
- `damage_dropout_curves.png`
- `growth_strips.png`
- `damage_circle_strips.png`
- `damage_half_strips.png`
- `damage_dropout_strips.png`
