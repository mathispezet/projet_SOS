"""End-to-end smoke test: a tiny regenerate-regime training and its evaluation on the CPU."""
import pathlib
import tempfile
import unittest

from nca import evaluate, train


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


if __name__ == "__main__":
    unittest.main()
