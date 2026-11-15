"""Compare several runs side by side: markdown tables, overlaid MSE curves and stacked strips.

usage:
  python scripts/compare_runs.py results/lizard_grow_s0 results/lizard_persist_s0 results/lizard_regen_s0 \
      --out results/ablation_A_regime [--labels grow persist regenerate] [--bands]
  python scripts/compare_runs.py "results/lizard_grow_s*" "results/lizard_persist_s*" "results/lizard_regen_s*" \
      --out results/ablation_A_seeds --by-condition --bands

Reads <run>/config.json, <run>/summary.json, <run>/eval/metrics.json and, when present,
<run>/eval/curves.npz and the strips written by ``nca.evaluate``. Writes to <out>:
comparison.md, growth_curves.png, growth_strips.png, damage_<kind>_curves.png and
damage_<kind>_strips.png (one row per run). Quoted glob patterns are expanded by the script.

With ``--by-condition`` the runs that share a label (by default the regime plus the non-default
settings, i.e. one condition trained with several seeds) are pooled: the tables give the mean over
the seeds with the [min, max] range, the curves average the trials of all seeds and ``--bands``
shades the min-max band of the per-seed mean curves (for a single run: the band of its trials).
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
Group = tuple[str, list[dict]]  # label, runs sharing it


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(v, spec: str = ".3g") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return format(v, spec)


def agg(values, spec: str = ".3g") -> str:
    """One value as is; several as ``mean [min, max]``."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "-"
    if len(vals) == 1:
        return fmt(vals[0], spec)
    return f"{fmt(sum(vals) / len(vals), spec)} [{fmt(min(vals), spec)}, {fmt(max(vals), spec)}]"


def default_label(config: dict) -> str:
    """Regime, plus the non-default ablation settings (fire rate, perception)."""
    extras = []
    if config.get("fire_rate", 0.5) != 0.5:
        extras.append(f"p={config['fire_rate']}")
    if config.get("perception", "sobel") != "sobel":
        extras.append(config["perception"])
    return config["regime"] + (f" ({', '.join(extras)})" if extras else "")


def expand(patterns: list[str]) -> list[Path]:
    """Run directories, expanding glob patterns (PowerShell and cmd do not)."""
    paths: list[Path] = []
    for pat in patterns:
        if any(c in pat for c in "*?["):
            matches = sorted(p for p in Path().glob(pat) if p.is_dir())
            if not matches:
                raise SystemExit(f"{pat}: no run directory matches")
            paths += matches
        else:
            paths.append(Path(pat))
    return paths


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


def group_runs(runs: list[dict], labels: list[str]) -> list[Group]:
    groups: dict[str, list[dict]] = {}
    for r, lab in zip(runs, labels):
        groups.setdefault(lab, []).append(r)
    return list(groups.items())


def is_multi(groups: list[Group]) -> bool:
    return any(len(rs) > 1 for _, rs in groups)


def table(cols: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows]


def growth_table(groups: list[Group]) -> list[str]:
    multi = is_multi(groups)
    cols = (["condition", "seeds"] if multi else ["run"]) + ["train loss (last 100 it.)", "MSE t=100", "MSE t=500",
                                                             "MSE t=1000"] + ([] if multi else ["training (min)"])
    rows = []
    for lab, rs in groups:
        s, m = [r["summary"] for r in rs], [r["metrics"] for r in rs]
        row = [lab] + ([str(len(rs))] if multi else [])
        row += [agg(x.get("mean_last_100") for x in s), agg(x.get("mse_at_100") for x in m),
                agg(x.get("mse_at_500") for x in m), agg(x.get("mse_at_1000") for x in m)]
        if not multi:
            row.append(agg((x.get("train_minutes") for x in s), ".1f"))
        rows.append(row)
    return table(cols, rows)


def damage_table(groups: list[Group]) -> list[str]:
    multi = is_multi(groups)
    if multi:
        cols = ["condition", "damage", "seeds", "MSE at end of window", "residual ratio", "recovery rate",
                "mean recovery time (steps)"]
    else:
        cols = ["run", "damage", "MSE after damage", "MSE at end of window", "undamaged control", "residual ratio",
                "recovery rate", "mean recovery time (steps)"]
    rows = []
    for lab, rs in groups:
        for kind in DAMAGE_KINDS:
            ds = [r["metrics"][f"damage_{kind}"] for r in rs if f"damage_{kind}" in r["metrics"]]
            if not ds:
                continue
            row = [lab, kind]
            if multi:
                row.append(str(len(ds)))
            else:
                row.append(agg(d.get("post_damage_mse") for d in ds))
            row.append(agg(d.get("final_mse") for d in ds))
            if not multi:
                row.append(agg(d.get("undamaged_mse_at_end") for d in ds))
            row += [agg(d.get("residual_ratio") for d in ds), agg((d.get("recovery_rate") for d in ds), ".2f"),
                    agg((d.get("mean_recovery_time") for d in ds), ".0f")]
            rows.append(row)
    return table(cols, rows)


def protocol_line(groups: list[Group]) -> str:
    runs = [r for _, rs in groups for r in rs]
    for r in runs:
        m = r["metrics"]
        if "damage_at" in m:
            text = (f"Damage applied at step {m['damage_at']}, then {m['steps_after']} steps; {m['trials']} trials per "
                    f"condition. The undamaged control is the mean MSE of intact growths at the same step and the "
                    f"residual ratio is the MSE at the end of the window divided by that control. A trial counts as "
                    f"recovered when its MSE is under {m['threshold']} and within {m.get('rel_factor', '-')}x of the "
                    f"control (floor {m.get('floor', '-')}) at the end of the window and stayed there; the recovery "
                    f"time is the first step of that sustained return.")
            if is_multi(groups):
                seeds = sorted({r["config"].get("seed") for r in runs})
                text += (f" Each condition pools its training seeds ({', '.join(map(str, seeds))}): the cells give "
                         f"the mean over the seeds with the [min, max] range; a recovery rate is therefore averaged "
                         f"over seeds x trials.")
            return text
    return ""


def curve_figures(groups: list[Group], out: Path, bands: bool) -> list[str]:
    written: list[str] = []

    def overlay(key: str, name: str, title: str, vlines=None):
        means, band = {}, {}
        for lab, rs in groups:
            per_run = [r["curves"][key] for r in rs if r["curves"] is not None and key in r["curves"]]
            if not per_run:
                continue
            n = min(c.shape[1] for c in per_run)
            pooled = np.concatenate([c[:, :n] for c in per_run], axis=0)  # (runs x trials, steps)
            means[lab] = pooled.mean(axis=0)
            if bands and len(per_run) > 1:
                seed_means = np.stack([c[:, :n].mean(axis=0) for c in per_run])
                band[lab] = (seed_means.min(axis=0), seed_means.max(axis=0))
            elif bands:
                band[lab] = (pooled.min(axis=0), pooled.max(axis=0))
        if not means:
            return
        plot_curves(out / name, means, title=title, vlines=vlines, bands=band or None)
        written.append(name)

    overlay("growth", "growth_curves.png", "growth from the seed (mean MSE over trials)")
    damage_at = next((int(r["curves"]["damage_at"]) for _, rs in groups for r in rs
                      if r["curves"] is not None and "damage_at" in r["curves"]), None)
    for kind in DAMAGE_KINDS:
        overlay(f"damage_{kind}", f"damage_{kind}_curves.png",
                f"{kind} damage at step {damage_at} (mean MSE over trials)", vlines=[damage_at] if damage_at else None)
    return written


def stack_strips(paths: list[Path], labels: list[str], out: Path) -> bool:
    """Stack the runs' strips vertically, one labelled row per run."""
    rows = [(Image.open(p).convert("RGB"), lab) for p, lab in zip(paths, labels) if p.exists()]
    if not rows:
        return False
    gap = 6
    label_w = max(110, int(max(ImageDraw.Draw(rows[0][0]).textlength(lab) for _, lab in rows)) + 10)
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
    p.add_argument("runs", nargs="+", help="run or results directories (quoted glob patterns allowed)")
    p.add_argument("--labels", nargs="*", default=None, help="one label per run (default: regime + non-default settings)")
    p.add_argument("--out", default="results/comparison")
    p.add_argument("--by-condition", action="store_true",
                   help="pool the runs that share a label (same condition, several training seeds)")
    p.add_argument("--bands", action="store_true", help="shade the min-max band of each curve (see module docstring)")
    args = p.parse_args(argv)

    runs = [load_run(path) for path in expand(args.runs)]
    labels = args.labels or [default_label(r["config"]) for r in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels needs one entry per run")
    if args.by_condition:
        groups = group_runs(runs, labels)
    else:
        if len(set(labels)) != len(labels):
            labels = [r["dir"].name for r in runs]
        groups = [(lab, [r]) for r, lab in zip(runs, labels)]
    multi = is_multi(groups)
    row_runs = [r for _, rs in groups for r in rs]
    row_labels = [f"{lab} s{r['config'].get('seed', '?')}" if multi else lab for lab, rs in groups for r in rs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    figures = curve_figures(groups, out, args.bands)
    if stack_strips([r["dir"] / "eval" / "growth_strip.png" for r in row_runs], row_labels, out / "growth_strips.png"):
        figures.append("growth_strips.png")
    for kind in DAMAGE_KINDS:
        name = f"damage_{kind}_strips.png"
        if stack_strips([r["dir"] / "eval" / f"damage_{kind}_strip.png" for r in row_runs], row_labels, out / name):
            figures.append(name)

    md = ["# Run comparison", "", "Runs: " + ", ".join(f"`{r['dir'].as_posix()}`" for r in runs), "",
          "## Growth and long-horizon stability", "", *growth_table(groups), "",
          "## Damage recovery", ""]
    proto = protocol_line(groups)
    if proto:
        md += [proto, ""]
    md += [*damage_table(groups), "", "## Figures", "", *[f"- `{f}`" for f in figures]]
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
