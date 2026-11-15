"""End-to-end smoke test: a tiny regenerate-regime training and its evaluation on the CPU."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

from nca import evaluate, train

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestTrainEvaluateSmoke(unittest.TestCase):
    def test_regenerate_regime_then_evaluate_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "run"
            summary = train.main([
                "--target", "disc", "--target-size", "16", "--pad", "4",
                "--regime", "regenerate", "--steps", "3", "--batch-size", "4", "--pool-size", "8",
                "--damage-n", "1", "--min-steps", "4", "--max-steps", "6",
                "--device", "cpu", "--log-every", "1", "--ckpt-every", "2", "--out", str(out),
            ])
            self.assertIn("final_loss", summary)
            self.assertTrue((out / "model.pt").exists())
            self.assertTrue((out / "growth_strip.png").exists())

            metrics = evaluate.main([
                "--run", str(out), "--steps", "20", "--trials", "2", "--damage", "all",
                "--damage-at", "5", "--steps-after", "5", "--device", "cpu",
            ])
            for kind in ("circle", "half", "dropout"):
                self.assertIn("recovery_rate", metrics[f"damage_{kind}"])
            self.assertTrue((out / "eval" / "metrics.json").exists())
            self.assertTrue((out / "eval" / "curves.npz").exists())

            cmp_out = pathlib.Path(tmp) / "cmp"
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), str(out),
                                   "--out", str(cmp_out)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((cmp_out / "comparison.md").exists())
            self.assertTrue((cmp_out / "growth_curves.png").exists())
            self.assertTrue((cmp_out / "damage_circle_curves.png").exists())

            # the same run twice = one condition with two "seeds": exercises the pooled tables and bands
            grp_out = pathlib.Path(tmp) / "grouped"
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), str(out), str(out),
                                   "--out", str(grp_out), "--by-condition", "--bands"],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            md = (grp_out / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("| condition | seeds |", md)
            self.assertIn("| regenerate | 2 |", md)
            self.assertTrue((grp_out / "damage_half_curves.png").exists())
            self.assertTrue((grp_out / "damage_half_strips.png").exists())


if __name__ == "__main__":
    unittest.main()
