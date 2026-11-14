"""Train a Neural Cellular Automaton to grow (and optionally persist / regenerate) a target.

Regimes (ablation A):
  grow        every run starts from the seed                     (Distill experiment 1)
  persist     batches come from a sample pool of end states      (Distill experiment 2)
  regenerate  persist + circular damage on the best pool samples (Distill experiment 3)

Example:
  python -m nca.train --target lizard --regime regenerate --steps 8000 --out runs/lizard_regen
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .evaluate import make_growth_strip
from .model import CAModel
from .pool import SamplePool
from .targets import make_seed, per_sample_mse, prepare_target, target_to_tensor
from .utils import make_circle_masks, plot_curves, resolve_device, set_seed, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="lizard", help="emoji name (lizard, mushroom, smiley, ...), 'disc' or a PNG path")
    p.add_argument("--target-size", type=int, default=40)
    p.add_argument("--pad", type=int, default=16)
    p.add_argument("--cache-dir", default="assets/targets")
    p.add_argument("--regime", choices=["grow", "persist", "regenerate"], default="grow")
    p.add_argument("--steps", type=int, default=8000, help="training iterations")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=1024)
    p.add_argument("--damage-n", type=int, default=3, help="damaged samples per batch (regenerate regime)")
    p.add_argument("--min-steps", type=int, default=64, help="CA steps per iteration, lower bound (inclusive)")
    p.add_argument("--max-steps", type=int, default=96, help="CA steps per iteration, upper bound (exclusive)")
    p.add_argument("--channel-n", type=int, default=16)
    p.add_argument("--hidden-n", type=int, default=128)
    p.add_argument("--fire-rate", type=float, default=0.5, help="ablation B: cell update probability")
    p.add_argument("--perception", choices=["sobel", "sobel_lap", "learned"], default="sobel", help="ablation C")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-decay-step", type=int, default=2000, help="iteration after which lr is multiplied by 0.1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute CA steps in the backward pass (less GPU memory, ~30%% slower)")
    p.add_argument("--out", required=True, help="output directory (config, checkpoints, logs, figures)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_np = prepare_target(args.target, args.target_size, args.pad, args.cache_dir)
    h, w = target_np.shape[:2]
    target = target_to_tensor(target_np, device)
    seed = make_seed(h, w, args.channel_n)
    seed_dev = seed.to(device)

    model = CAModel(args.channel_n, args.hidden_n, args.fire_rate, args.perception).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_decay_step], gamma=0.1)
    pool = SamplePool(seed, args.pool_size) if args.regime != "grow" else None

    config = {**vars(args), "grid": [h, w], "device": str(device), "model": model.describe()}
    write_json(out / "config.json", config)
    print(f"[train] {args.regime} on '{args.target}' {h}x{w}, {config['model']['parameters']} params, device {device}")

    log_path = out / "log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    logger.writerow(["step", "loss", "lr", "ca_steps", "elapsed_s"])
    losses: list[float] = []
    t0 = time.time()

    for it in range(1, args.steps + 1):
        if pool is None:
            idx = None
            x0 = seed_dev.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)
        else:
            idx, x0 = pool.sample(args.batch_size)
            x0 = x0.to(device)
            with torch.no_grad():
                rank = per_sample_mse(x0, target).argsort(descending=True)
            x0, idx = x0[rank], idx[rank]
            x0[:1] = seed_dev  # the worst sample restarts from the seed
            if args.regime == "regenerate" and args.damage_n > 0:
                masks = make_circle_masks(args.damage_n, h, w, device=device)
                x0[-args.damage_n :] = x0[-args.damage_n :] * (1.0 - masks).unsqueeze(1)

        ca_steps = random.randint(args.min_steps, args.max_steps - 1)
        if args.grad_checkpoint:
            x = x0
            for _ in range(ca_steps):  # RNG state is preserved, so the same update mask is replayed
                x = checkpoint(model, x, use_reentrant=False)
        else:
            x = model.run(x0, ca_steps)
        loss = per_sample_mse(x, target).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():  # per-parameter gradient normalisation (Distill)
                if p.grad is not None:
                    p.grad.div_(p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        if pool is not None:
            pool.commit(idx, x)

        loss_v = float(loss.item())
        losses.append(loss_v)
        elapsed = time.time() - t0
        logger.writerow([it, f"{loss_v:.6f}", f"{opt.param_groups[0]['lr']:.6g}", ca_steps, f"{elapsed:.1f}"])
        if it % args.log_every == 0 or it == 1:
            recent = float(np.mean(losses[-args.log_every :]))
            eta = elapsed / it * (args.steps - it)
            print(f"[train] step {it:5d}/{args.steps}  loss {loss_v:.5f}  (avg {recent:.5f})  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {elapsed/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
            log_file.flush()
        if it % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "config": config, "step": it}, out / "model.pt")

    log_file.close()
    torch.save({"model": model.state_dict(), "config": config, "step": args.steps}, out / "model.pt")
    plot_curves(out / "loss.png", {"train loss": np.array(losses)}, xlabel="training iteration",
                ylabel="loss (MSE)", title=f"{args.regime} / {args.target}")
    strip = make_growth_strip(model, seed_dev, device)
    strip.save(out / "growth_strip.png")
    summary = {
        "final_loss": losses[-1],
        "best_loss": float(min(losses)),
        "mean_last_100": float(np.mean(losses[-100:])),
        "train_minutes": (time.time() - t0) / 60.0,
        "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
    }
    write_json(out / "summary.json", summary)
    print(f"[train] done: final loss {summary['final_loss']:.5f}, best {summary['best_loss']:.5f}, "
          f"{summary['train_minutes']:.1f} min -> {out}")
    return summary


if __name__ == "__main__":
    main()
