"""Evaluate a trained NCA: growth curve, long-horizon stability and damage recovery.

Example:
  python -m nca.evaluate --run runs/lizard_regen --damage all --trials 20
Outputs go to <run>/eval/: metrics.json, curves.npz (per-trial MSE curves, for cross-run
figures), growth_curve.png, growth_strip.png and one curve + one strip per damage type
(circle, half, dropout).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import CAModel
from .targets import make_seed, per_sample_mse, prepare_target, target_to_tensor, to_rgb
from .utils import (
    cell_dropout,
    circle_mask_at,
    erase_half,
    frames_to_strip,
    plot_curves,
    read_json,
    resolve_device,
    set_seed,
    tensor_to_uint8,
    write_json,
)

DAMAGE_TYPES = ("circle", "half", "dropout")


def load_run(run_dir: str | Path, device: torch.device | str):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    model = CAModel(config["channel_n"], config["hidden_n"], config["fire_rate"], config["perception"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    target_np = prepare_target(config["target"], config["target_size"], config["pad"], config.get("cache_dir", "assets/targets"))
    target = target_to_tensor(target_np, device)
    seed = make_seed(*target_np.shape[:2], config["channel_n"]).to(device)
    return model, config, target, seed


@torch.no_grad()
def make_growth_strip(model: CAModel, seed: torch.Tensor, device, snapshots=(0, 8, 16, 32, 64, 96, 128, 200)) -> Image.Image:
    """Frames of one growth from the seed at the given CA steps."""
    x = seed.unsqueeze(0).to(device)
    frames, labels = [], []
    for step in range(max(snapshots) + 1):
        if step in snapshots:
            frames.append(tensor_to_uint8(to_rgb(x)[0]))
            labels.append(f"t={step}")
        x = model(x)
    return frames_to_strip(frames, labels=labels)


@torch.no_grad()
def growth_curve(model: CAModel, seed: torch.Tensor, target: torch.Tensor, steps: int, trials: int) -> np.ndarray:
    """(trials, steps + 1) MSE to the target at every step of a growth from the seed."""
    x = seed.unsqueeze(0).repeat(trials, 1, 1, 1)
    curve = [per_sample_mse(x, target).cpu().numpy()]
    for _ in range(steps):
        x = model(x)
        curve.append(per_sample_mse(x, target).cpu().numpy())
    return np.stack(curve, axis=1)


def apply_damage(x: torch.Tensor, kind: str, radius: float, dropout_p: float, rng: torch.Generator) -> torch.Tensor:
    """Damage every state in the batch independently (same type, random placement)."""
    if kind == "circle":
        h, w = x.shape[-2:]
        masks = []
        for _ in range(x.shape[0]):
            cx, cy = (torch.rand(2, generator=rng) - 0.5).tolist()
            masks.append(circle_mask_at(h, w, cx, cy, radius, device=x.device))
        return x * (1.0 - torch.cat(masks, dim=0)).unsqueeze(1)
    if kind == "half":
        sides = ["right", "left", "bottom", "top"]
        return torch.cat([erase_half(x[i : i + 1], sides[int(torch.randint(0, 4, (1,), generator=rng))])
                          for i in range(x.shape[0])], dim=0)
    if kind == "dropout":
        return cell_dropout(x, dropout_p)
    raise ValueError(kind)


@torch.no_grad()
def damage_experiment(model, seed, target, device, kind, damage_at, steps_after, radius, dropout_p, trials,
                      threshold, rng, control=None, rel_factor=2.0, floor=1e-3):
    """Grow for ``damage_at`` steps, damage, then run ``steps_after`` more steps.

    ``control`` is the mean MSE curve of undamaged growths, indexed by absolute CA step. A trial counts
    as recovered at a step when its MSE is both under the absolute quality ``threshold`` and within
    ``rel_factor`` times the control at the same step (that relative target is never stricter than
    ``floor``), and stays so until the end of the window; the recovery time is the first step of that
    sustained return. Returns the per-trial MSE curves (trials, damage_at + steps_after + 1), the
    recovery stats and strip frames."""
    x = seed.unsqueeze(0).repeat(trials, 1, 1, 1)
    curve = [per_sample_mse(x, target).cpu().numpy()]
    for _ in range(damage_at):
        x = model(x)
        curve.append(per_sample_mse(x, target).cpu().numpy())
    pre_mse = curve[-1].copy()
    x = apply_damage(x, kind, radius, dropout_p, rng)
    frames = [tensor_to_uint8(to_rgb(x)[0])]
    labels = ["damaged"]
    snap = {1, 8, 32, 64, 128, steps_after}
    for t in range(1, steps_after + 1):
        x = model(x)
        curve.append(per_sample_mse(x, target).cpu().numpy())
        if t in snap:
            frames.append(tensor_to_uint8(to_rgb(x)[0]))
            labels.append(f"+{t}")
    curve = np.stack(curve, axis=1)
    after = curve[:, damage_at + 1 :]  # (trials, steps_after); after[:, t - 1] is t CA steps after the damage
    steps = np.arange(damage_at + 1, damage_at + steps_after + 1)  # absolute CA step of each column
    if control is None:
        targets = np.full(steps_after, threshold)
    else:
        targets = np.minimum(threshold, np.maximum(rel_factor * np.asarray(control)[steps], floor))
    ok = after <= targets[None, :]
    sustained = np.flip(np.cumprod(np.flip(ok, axis=1), axis=1), axis=1).astype(bool)
    recovered = ok[:, -1]
    rec_time = np.where(recovered, sustained.argmax(axis=1) + 1, np.nan)
    final_mse = float(after[:, -1].mean())
    end_control = float(control[steps[-1]]) if control is not None else None
    stats = {
        "pre_damage_mse": float(pre_mse.mean()),
        "post_damage_mse": float(after[:, 0].mean()),  # one CA step after the damage
        "final_mse": final_mse,
        "undamaged_mse_at_end": end_control,
        "residual_ratio": (final_mse / end_control) if end_control else None,
        "target_at_end": float(targets[-1]),
        "recovery_rate": float(recovered.mean()),
        "mean_recovery_time": float(np.nanmean(rec_time)) if recovered.any() else None,
        "trials": trials,
    }
    return curve, stats, frames_to_strip(frames, labels=labels)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training output directory")
    p.add_argument("--out", default=None, help="defaults to <run>/eval")
    p.add_argument("--steps", type=int, default=1000, help="growth length for the stability curve")
    p.add_argument("--damage", default="all", help="none, all, or a comma list of: circle, half, dropout")
    p.add_argument("--damage-at", type=int, default=100)
    p.add_argument("--steps-after", type=int, default=200)
    p.add_argument("--radius", type=float, default=0.3, help="circle radius in [-1, 1] coordinates")
    p.add_argument("--dropout-p", type=float, default=0.3)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.01,
                   help="absolute MSE a damaged run must get back under to count as recovered")
    p.add_argument("--rel-factor", type=float, default=2.0,
                   help="... and be within this factor of the undamaged control at the same step")
    p.add_argument("--floor", type=float, default=1e-3, help="the relative target is never stricter than this MSE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, config, target, seed = load_run(args.run, device)
    out = Path(args.out) if args.out else Path(args.run) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(args.seed)

    metrics: dict = {"run": str(args.run), "regime": config["regime"], "target": config["target"],
                     "trials": args.trials, "damage_at": args.damage_at, "steps_after": args.steps_after,
                     "threshold": args.threshold, "rel_factor": args.rel_factor, "floor": args.floor}
    n_steps = max(args.steps, args.damage_at + args.steps_after)  # the control must cover the damage window
    curves = growth_curve(model, seed, target, n_steps, args.trials)
    saved_curves = {"growth": curves.astype(np.float32)}  # raw per-trial curves, for cross-run figures
    mean = curves.mean(axis=0)
    for t in (50, 100, 200, 500, 1000):
        if t <= n_steps:
            metrics[f"mse_at_{t}"] = float(mean[t])
    metrics["mse_min"] = float(mean.min())
    metrics["mse_min_step"] = int(mean.argmin())
    plot_curves(out / "growth_curve.png", {"growth": mean}, title="growth from the seed",
                bands={"growth": (curves.min(axis=0), curves.max(axis=0))})
    make_growth_strip(model, seed, device).save(out / "growth_strip.png")

    kinds = [] if args.damage == "none" else (list(DAMAGE_TYPES) if args.damage == "all" else args.damage.split(","))
    for kind in kinds:
        curve, stats, strip = damage_experiment(model, seed, target, device, kind, args.damage_at, args.steps_after,
                                                args.radius, args.dropout_p, args.trials, args.threshold, rng,
                                                control=mean, rel_factor=args.rel_factor, floor=args.floor)
        saved_curves[f"damage_{kind}"] = curve.astype(np.float32)
        metrics[f"damage_{kind}"] = stats
        m = curve.mean(axis=0)
        plot_curves(out / f"damage_{kind}_curve.png", {kind: m}, vlines=[args.damage_at],
                    title=f"{kind} damage at step {args.damage_at}", bands={kind: (curve.min(axis=0), curve.max(axis=0))})
        strip.save(out / f"damage_{kind}_strip.png")
        print(f"[eval] {kind:8s} recovery rate {stats['recovery_rate']:.2f}  "
              f"pre {stats['pre_damage_mse']:.4f} -> post {stats['post_damage_mse']:.4f} -> final {stats['final_mse']:.4f} "
              f"(undamaged control {stats['undamaged_mse_at_end']:.4f}, residual ratio {stats['residual_ratio']:.2f})")

    np.savez_compressed(out / "curves.npz", damage_at=np.int64(args.damage_at), **saved_curves)
    write_json(out / "metrics.json", metrics)
    print(f"[eval] growth MSE: " + ", ".join(f"t={t}: {metrics[f'mse_at_{t}']:.4f}" for t in (50, 100, 200, 500, 1000) if f"mse_at_{t}" in metrics))
    print(f"[eval] wrote {out}")
    return metrics


if __name__ == "__main__":
    main()
