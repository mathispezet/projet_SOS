# NCA self-repair: learning local rules for growth and regeneration

Project for the course **Self-Organizing Systems** (University POLITEHNICA of Bucharest,
Faculty of Automatic Control and Computers, 2026-2027).

A **Neural Cellular Automaton** (NCA) is a grid of identical cells. Each cell sees only its
3x3 neighbourhood and applies an update rule that is a small neural network, trained by
gradient descent so that, starting from a single seed cell, the grid grows a target image,
keeps it stable and regenerates it after damage. There is no central controller and no
global clock: the global pattern is emergent.

The project reimplements *Growing Neural Cellular Automata* (Mordvintsev et al., Distill
2020) in PyTorch and studies **self-repair quantitatively**:

- **RQ1** — can a learned local rule reliably produce and maintain a global pattern?
- **RQ2** — which choices determine self-repair: the training regime, the stochastic update
  rate (asynchrony) and the perception kernels?

## Quickstart (Windows, NVIDIA GPU)

```bash
python -m venv .venv
.venv\Scripts\activate
# PyTorch with CUDA 12.8 (required for RTX 50-series GPUs); CPU-only also works, slowly
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Train, then evaluate:

```bash
python -m nca.train --target lizard --regime grow --steps 8000 --out runs/lizard_grow
python -m nca.evaluate --run runs/lizard_grow --damage all --trials 20
python scripts/collect_results.py runs/lizard_grow          # copy metrics and figures into results/
python scripts/compare_runs.py results/lizard_grow results/lizard_regen --out results/comparison
```

The whole experiment (7 conditions x 3 seeds, about 16 min of GPU per run) is one command, which
skips the runs that already exist and can be resumed after an interruption; pooled tables over
the seeds of each condition come from `compare_runs.py --by-condition`:

```bash
python scripts/run_ablations.py --seeds 0 1 2
python scripts/compare_runs.py "results/lizard_grow_s*" "results/lizard_persist_s*" "results/lizard_regen_s[0-9]" \
    --out results/ablation_A_seeds --by-condition --bands
```

Targets are Noto emoji downloaded on first use (`lizard`, `mushroom`, `smiley`, `butterfly`,
`fish`, `star`), the offline synthetic `disc`, or any PNG path.

## Experiments

| Ablation | Flag | Values |
|---|---|---|
| A. Training regime | `--regime` | `grow` (seed only), `persist` (+ sample pool), `regenerate` (+ damage during training) |
| B. Update probability | `--fire-rate` | 1.0, 0.5, 0.25 |
| C. Perception | `--perception` | `sobel`, `sobel_lap`, `learned` |

Evaluation (`nca.evaluate`): MSE to the target at every step of a growth from the seed
(long-horizon stability at steps 500 and 1000), and three damage types applied at step 100
(circular erasure, half cut, random cell dropout) with the **recovery rate** (share of runs
whose MSE, 200 steps after the damage, is back under 0.01 and within 2x of an undamaged
growth at the same step), the **recovery time** and the **residual ratio** (MSE at the end of
that window divided by the undamaged control), over 3 training seeds x 20 damage trials per
condition. Each evaluation also keeps the raw per-trial curves
(`eval/curves.npz`) so that `scripts/compare_runs.py` can overlay several runs and tabulate
them. Results are in `results/` and discussed in `RESULTS.md`; the written report is
`report/SOS_report_NCA_self_repair.pdf` and the final presentation `report/SOS_final_presentation_NCA.pptx`.

## Repository layout

```
nca/model.py      CAModel: perception, update rule, stochastic update, alive masking
nca/targets.py    emoji / synthetic targets, seed, colour conversions, MSE
nca/pool.py       sample pool for persistence training
nca/train.py      training loop (regimes, gradient normalisation, logging, figures)
nca/evaluate.py   growth curves, damage protocol, recovery metrics, strips
nca/utils.py      damage masks, plotting, seeding
scripts/          run_ablations.py (train + evaluate + collect every condition and seed),
                  collect_results.py (copy a run into results/), compare_runs.py (tables + overlaid curves)
tests/            unit tests (unittest)
runs/             training outputs (ignored by git)
results/          versioned metrics and figures per run, cross-run comparisons
report/           the written project report (PDF) and the final presentation (PPTX)
```

## References

- Mordvintsev, A., Randazzo, E., Niklasson, E., Levin, M. (2020). Growing Neural Cellular Automata. *Distill*. https://distill.pub/2020/growing-ca/
- Gilpin, W. (2019). Cellular automata as convolutional neural networks. *Physical Review E*, 100(3), 032402.
- Randazzo, E., Mordvintsev, A., Niklasson, E., Levin, M., Greydanus, S. (2020). Self-classifying MNIST Digits. *Distill*. https://distill.pub/2020/selforg/mnist/
- Niklasson, E., Mordvintsev, A., Randazzo, E., Levin, M. (2021). Self-Organising Textures. *Distill*. https://distill.pub/selforg/2021/textures/

## License

MIT, see `LICENSE`. Emoji targets are from Noto Emoji (Google, Apache 2.0).
