"""Unit tests (standard library unittest; also runs under pytest).

    python -m unittest discover -s tests -v
"""
import unittest

import torch

from nca.model import CAModel
from nca.pool import SamplePool
from nca.targets import make_seed, per_sample_mse, prepare_target, target_to_tensor, to_rgb
from nca.utils import cell_dropout, erase_half, make_circle_masks


class TestModel(unittest.TestCase):
    def test_zero_init_leaves_seed_unchanged(self):
        torch.manual_seed(0)
        model = CAModel()
        x = make_seed(24, 24).unsqueeze(0)
        y = model(x)
        self.assertTrue(torch.equal(x, y))

    def test_perceive_shape(self):
        model = CAModel(perception="sobel_lap")
        x = torch.rand(2, 16, 20, 20)
        self.assertEqual(tuple(model.perceive(x).shape), (2, 16 * 4, 20, 20))

    def test_perceive_identity_channel_is_input(self):
        model = CAModel(perception="sobel")
        x = torch.rand(1, 16, 12, 12)
        p = model.perceive(x)
        # output channel c*K + 0 is the identity filter applied to input channel c
        self.assertTrue(torch.allclose(p[:, 0::3], x, atol=1e-6))

    def test_alive_masking_kills_isolated_immature_cell(self):
        model = CAModel()
        x = torch.zeros(1, 16, 10, 10)
        x[0, 3, 5, 5] = 0.05  # below the alive threshold, no mature neighbour
        with torch.no_grad():
            y = model(x)
        self.assertEqual(float(y.abs().sum()), 0.0)

    def test_learned_perception_is_a_parameter(self):
        names = [n for n, _ in CAModel(perception="learned").named_parameters()]
        self.assertIn("filters", names)
        self.assertNotIn("filters", [n for n, _ in CAModel(perception="sobel").named_parameters()])

    def test_gradients_flow_through_steps(self):
        torch.manual_seed(0)
        model = CAModel()
        x = make_seed(16, 16).unsqueeze(0)
        y = model.run(x, 4)
        loss = (y[:, :4] ** 2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_fire_rate_one_updates_every_cell(self):
        torch.manual_seed(0)
        model = CAModel(fire_rate=1.0)
        with torch.no_grad():
            model.rule[2].weight.normal_(0, 0.1)
        x = torch.rand(1, 16, 8, 8) + 0.5  # every cell mature
        y = model(x)
        self.assertFalse(torch.equal(x, y))


class TestPool(unittest.TestCase):
    def test_sample_and_commit(self):
        pool = SamplePool(make_seed(8, 8), size=16)
        idx, x = pool.sample(4)
        self.assertEqual(tuple(x.shape), (4, 16, 8, 8))
        pool.commit(idx, x + 1.0)
        self.assertTrue(torch.equal(pool.x[idx], x + 1.0))


class TestTargets(unittest.TestCase):
    def test_synthetic_target_is_padded_and_premultiplied(self):
        t = prepare_target("disc", size=40, pad=16)
        self.assertEqual(t.shape, (72, 72, 4))
        self.assertTrue((t[..., :3] <= t[..., 3:] + 1e-6).all())
        self.assertEqual(float(t[:16].sum()), 0.0)  # padding is empty

    def test_mse_and_rgb(self):
        t = target_to_tensor(prepare_target("disc"))
        x = torch.zeros(2, 16, 72, 72)
        x[:, :4] = t
        self.assertTrue(torch.allclose(per_sample_mse(x, t), torch.zeros(2)))
        rgb = to_rgb(x)
        self.assertEqual(tuple(rgb.shape), (2, 3, 72, 72))
        self.assertTrue(float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0)


class TestDamage(unittest.TestCase):
    def test_circle_masks(self):
        m = make_circle_masks(5, 72, 72)
        self.assertEqual(tuple(m.shape), (5, 72, 72))
        self.assertTrue(set(m.unique().tolist()) <= {0.0, 1.0})
        self.assertTrue(0.0 < float(m.mean()) < 0.5)

    def test_half_and_dropout(self):
        x = torch.ones(1, 16, 10, 10)
        self.assertEqual(float(erase_half(x, "right").sum()), 16 * 10 * 5)
        torch.manual_seed(0)
        self.assertLess(float(cell_dropout(x, 0.5).sum()), float(x.sum()))


if __name__ == "__main__":
    unittest.main()
