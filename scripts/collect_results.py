"""Copy the figures and metrics of one or more runs into the versioned ``results/`` folder.

``runs/`` is ignored by git (logs, intermediate checkpoints); ``results/`` keeps what the report
needs plus the final weights (``model.pt``, about 40 KB), so every published model can be
re-evaluated with ``python -m nca.evaluate --run results/<run> --out <dir>``.

usage: python scripts/collect_results.py runs/lizard_grow_s0 [runs/other_run ...]
"""
import json
import shutil
import sys
from pathlib import Path

KEEP = ["config.json", "summary.json", "model.pt", "loss.png", "growth_strip.png"]
KEEP_EVAL = ["metrics.json", "curves.npz", "growth_curve.png", "growth_strip.png",
             "damage_circle_curve.png", "damage_circle_strip.png",
             "damage_half_curve.png", "damage_half_strip.png",
             "damage_dropout_curve.png", "damage_dropout_strip.png"]


def collect(run: Path, dest_root: Path = Path("results")) -> Path:
    dest = dest_root / run.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (run / name).exists():
            shutil.copy2(run / name, dest / name)
    if (run / "eval").exists():
        (dest / "eval").mkdir(exist_ok=True)
        for name in KEEP_EVAL:
            if (run / "eval" / name).exists():
                shutil.copy2(run / "eval" / name, dest / "eval" / name)
    return dest


def one_line(run: Path) -> str:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    metrics_path = run / "eval" / "metrics.json"
    if not metrics_path.exists():
        return f"{run.name}: train loss {summary['final_loss']:.5f} (no eval yet)"
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    rates = ", ".join(f"{k}={m[f'damage_{k}']['recovery_rate']:.2f}" for k in ("circle", "half", "dropout")
                      if f"damage_{k}" in m)
    return (f"{run.name}: train loss {summary['final_loss']:.5f} | MSE t=100 {m.get('mse_at_100', float('nan')):.4f}, "
            f"t=500 {m.get('mse_at_500', float('nan')):.4f}, t=1000 {m.get('mse_at_1000', float('nan')):.4f} | "
            f"recovery {rates}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run = Path(arg)
        dest = collect(run)
        print(f"collected {run} -> {dest}")
        print("  " + one_line(run))
