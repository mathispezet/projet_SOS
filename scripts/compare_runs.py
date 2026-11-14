"""Compare several runs side by side: markdown tables, overlaid MSE curves and stacked strips.

usage:
  python scripts/compare_runs.py results/lizard_grow_s0 results/lizard_persist_s0 results/lizard_regen_s0 \
      --out results/ablation_A_regime [--labels grow persist regenerate] [--bands]

Reads <run>/config.json, <run>/summary.json, <run>/eval/metrics.json and, when present,
<run>/eval/curves.npz and the strips written by ``nca.evaluate``. Writes to <out>:
comparison.md, growth_curves.png, growth_strips.png, damage_<kind>_curves.png and
damage_<kind>_strips.png (one row per run).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so ``nca`` imports when run as a script
from nca.utils import plot_curves  # noqa: E402

DAMAGE_KINDS = ("circle", "half", "dropout")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def default_label(config: dict) -> str:
    """Regime, plus the non-default ablation settings (fire rate, perception)."""
    extras = []
    if config.get("fire_rate", 0.5) != 0.5:
        extras.append(f"p={config['fire_rate']}")
    if config.get("perception", "sobel") != "sobel":
        extras.append(config["perception"])
    return config["regime"] + (f" ({', '.join(extras)})" if extras else "")


def load_run(run: Path) -> dict:
    config = read_json(run / "config.json")
    if config is None:
        raise SystemExit(f"{run}: no config.json")
    curves = None
    npz = run / "eval" / "curves.npz"
    if npz.exists():
        with np.load(npz) as z:
            curves = {k: z[k] for k in z.files}
    return {
        "dir": run,
        "config": config,
        "summary": read_json(run / "summary.json") or {},
        "metrics": read_json(run / "eval" / "metrics.json") or {},
        "curves": curves,
    }


def growth_table(runs: list[dict], labels: list[str]) -> list[str]:
    lines = ["| run | train loss (last 100 it.) | MSE t=100 | MSE t=500 | MSE t=1000 | training (min) |",
             "|---|---|---|---|---|---|"]
    for r, lab in zip(runs, labels):
        s, m = r["summary"], r["metrics"]
        lines.append(f"| {lab} | {fmt(s.get('mean_last_100'), 5)} | {fmt(m.get('mse_at_100'))} | "
                     f"{fmt(m.get('mse_at_500'))} | {fmt(m.get('mse_at_1000'))} | {fmt(s.get('train_minutes'), 1)} |")
    return lines


def damage_table(runs: list[dict], labels: list[str]) -> list[str]:
    lines = ["| run | damage | MSE after damage | MSE at end of window | undamaged control | residual ratio | recovery rate | mean recovery time (steps) |",
             "|---|---|---|---|---|---|---|---|"]
    for r, lab in zip(runs, labels):
        for kind in DAMAGE_KINDS:
            d = r["metrics"].get(f"damage_{kind}")
            if d is None:
                continue
            lines.append(f"| {lab} | {kind} | {fmt(d.get('post_damage_mse'))} | {fmt(d.get('final_mse'))} | "
                         f"{fmt(d.get('undamaged_mse_at_end'))} | {fmt(d.get('residual_ratio'), 2)} | "
                         f"{fmt(d.get('recovery_rate'), 2)} | {fmt(d.get('mean_recovery_time'), 0)} |")
    return lines


def protocol_line(runs: list[dict]) -> str:
    for r in runs:
        m = r["metrics"]
        if "damage_at" in m:
            return (f"Damage applied at step {m['damage_at']}, then {m['steps_after']} steps; {m['trials']} trials per "
                    f"condition. The undamaged control is the mean MSE of intact growths at the same step and the "
                    f"residual ratio is the MSE at the end of the window divided by that control. A trial counts as "
                    f"recovered when its MSE is under {m['threshold']} and within {m.get('rel_factor', '-')}x of the "
                    f"control (floor {m.get('floor', '-')}) at the end of the window and stayed there; the recovery "
                    f"time is the first step of that sustained return.")
    return ""


def curve_figures(runs: list[dict], labels: list[str], out: Path, bands: bool) -> list[str]:
    written: list[str] = []
    sel = [(r, lab) for r, lab in zip(runs, labels) if r["curves"] is not None]
    if not sel:
        return written

    def overlay(key: str, name: str, title: str, vlines=None):
        have = [(r, lab) for r, lab in sel if key in r["curves"]]
        if not have:
            return
        means = {lab: r["curves"][key].mean(axis=0) for r, lab in have}
        band = {lab: (r["curves"][key].min(axis=0), r["curves"][key].max(axis=0)) for r, lab in have} if bands else None
        plot_curves(out / name, means, title=title, vlines=vlines, bands=band)
        written.append(name)

    overlay("growth", "growth_curves.png", "growth from the seed (mean MSE over trials)")
    damage_at = next((int(r["curves"]["damage_at"]) for r, _ in sel if "damage_at" in r["curves"]), None)
    for kind in DAMAGE_KINDS:
        overlay(f"damage_{kind}", f"damage_{kind}_curves.png",
                f"{kind} damage at step {damage_at} (mean MSE over trials)", vlines=[damage_at] if damage_at else None)
    return written


def stack_strips(paths: list[Path], labels: list[str], out: Path) -> bool:
    """Stack the runs' strips vertically, one labelled row per run."""
    rows = [(Image.open(p).convert("RGB"), lab) for p, lab in zip(paths, labels) if p.exists()]
    if not rows:
        return False
    label_w, gap = 110, 6
    width = label_w + max(im.width for im, _ in rows)
    height = sum(im.height for im, _ in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for im, lab in rows:
        draw.text((4, y + im.height // 2 - 5), lab, fill=(40, 40, 40))
        canvas.paste(im, (label_w, y))
        y += im.height + gap
    canvas.save(out)
    return True


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run or results directories")
    p.add_argument("--labels", nargs="*", default=None, help="one label per run (default: regime + non-default settings)")
    p.add_argument("--out", default="results/comparison")
    p.add_argument("--bands", action="store_true", help="shade the min-max band of each run in the overlaid curves")
    args = p.parse_args(argv)

    runs = [load_run(Path(r)) for r in args.runs]
    labels = args.labels or [default_label(r["config"]) for r in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels needs one entry per run")
    if len(set(labels)) != len(labels):
        labels = [r["dir"].name for r in runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    figures = curve_figures(runs, labels, out, args.bands)
    if stack_strips([r["dir"] / "eval" / "growth_strip.png" for r in runs], labels, out / "growth_strips.png"):
        figures.append("growth_strips.png")
    for kind in DAMAGE_KINDS:
        name = f"damage_{kind}_strips.png"
        if stack_strips([r["dir"] / "eval" / f"damage_{kind}_strip.png" for r in runs], labels, out / name):
            figures.append(name)

    md = ["# Run comparison", "", "Runs: " + ", ".join(f"`{r['dir'].as_posix()}`" for r in runs), "",
          "## Growth and long-horizon stability", "", *growth_table(runs, labels), "",
          "## Damage recovery", ""]
    proto = protocol_line(runs)
    if proto:
        md += [proto, ""]
    md += [*damage_table(runs, labels), "", "## Figures", "", *[f"- `{f}`" for f in figures]]
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
