"""Train, evaluate and collect every ablation condition for one or more training seeds.

usage:
  python scripts/run_ablations.py --seeds 1 2                       # the 7 conditions, seeds 1 and 2
  python scripts/run_ablations.py --seeds 0 --conditions grow regen  # a subset
  python scripts/run_ablations.py --seeds 1 2 --dry-run              # print the commands only

The 7 conditions cover the three ablations of RESULTS.md: A (grow / persist / regen), B (regen with
fire rate 1.0 / 0.25, the 0.5 baseline being ``regen``) and C (regen with Sobel+Laplacian / learned
perception). Runs are written to ``runs/<target>_<condition>_s<seed>`` (git-ignored) and copied into
``results/`` by ``collect_results.py``. A step whose output already exists is skipped, so the script
can be re-run after an interruption. Runs are processed seed by seed, so stopping the chain early
leaves complete seeds behind.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from collect_results import collect, one_line  # same directory as this script

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "grow": ["--regime", "grow"],
    "persist": ["--regime", "persist"],
    "regen": ["--regime", "regenerate"],
    "regen_p100": ["--regime", "regenerate", "--fire-rate", "1.0"],
    "regen_p025": ["--regime", "regenerate", "--fire-rate", "0.25"],
    "regen_soblap": ["--regime", "regenerate", "--perception", "sobel_lap"],
    "regen_learned": ["--regime", "regenerate", "--perception", "learned"],
}


def run_logged(cmd: list[str], log: Path, dry_run: bool) -> int:
    print("  $ " + " ".join(cmd) + f"  > {log.relative_to(ROOT).as_posix()}", flush=True)
    if dry_run:
        return 0
    with log.open("a", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def process(name: str, train_args: list[str], seed: int, args, python: str) -> bool:
    rel = f"runs/{name}"  # commands run with cwd=ROOT, so relative paths keep the log readable
    run_dir = ROOT / rel
    ok = True
    t0 = time.time()
    if (run_dir / "summary.json").exists():
        print(f"  training already done ({rel}/summary.json)")
    else:
        cmd = [python, "-m", "nca.train", "--target", args.target, *train_args, "--steps", str(args.steps),
               "--seed", str(seed), "--out", rel]
        ok = run_logged(cmd, run_dir.parent / f"{name}.log", args.dry_run) == 0
        if not ok:
            print(f"  TRAINING FAILED, see {rel}.log", flush=True)
    if ok and (run_dir / "eval" / "metrics.json").exists():
        print(f"  evaluation already done ({rel}/eval/metrics.json)")
    elif ok:
        cmd = [python, "-m", "nca.evaluate", "--run", rel, "--damage", "all", "--trials", str(args.trials)]
        ok = run_logged(cmd, run_dir.parent / f"{name}_eval.log", args.dry_run) == 0
        if not ok:
            print(f"  EVALUATION FAILED, see {rel}_eval.log", flush=True)
    if ok and not args.dry_run:
        dest = collect(run_dir, ROOT / "results")
        print(f"  collected -> {dest.relative_to(ROOT).as_posix()}")
        print("  " + one_line(run_dir))
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="training seeds, processed in this order")
    p.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), default=list(CONDITIONS))
    p.add_argument("--target", default="lizard")
    p.add_argument("--steps", type=int, default=8000, help="training iterations per run")
    p.add_argument("--trials", type=int, default=20, help="evaluation trials per damage type")
    p.add_argument("--python", default=sys.executable, help="interpreter used for training and evaluation")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running anything")
    args = p.parse_args(argv)

    conditions = [c for c in CONDITIONS if c in args.conditions]  # keep the canonical order
    jobs = [(seed, cond) for seed in args.seeds for cond in conditions]
    print(f"{len(jobs)} runs: {len(args.seeds)} seed(s) x {len(conditions)} condition(s), "
          f"{args.steps} iterations each", flush=True)
    failed = []
    start = time.time()
    for i, (seed, cond) in enumerate(jobs, 1):
        name = f"{args.target}_{cond}_s{seed}"
        print(f"\n[{i}/{len(jobs)}] {name}  ({time.strftime('%H:%M:%S')})", flush=True)
        if not process(name, CONDITIONS[cond], seed, args, args.python):
            failed.append(name)
    print(f"\nall done in {(time.time() - start) / 60:.1f} min; {len(failed)} failure(s)"
          + (": " + ", ".join(failed) if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
