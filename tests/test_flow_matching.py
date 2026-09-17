from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from tfm_shells.flow import integrate_flow, model_time
from tfm_shells.models.factory import build_unet
from tfm_shells.sampling.guided import SamplingContext, generate_samples
from tfm_shells.training.train_architect import _run_epoch as run_architect_epoch
from tfm_shells.training.train_engineer import _run_epoch


class FlowMatchingTests(unittest.TestCase):
    def test_constant_velocity_reaches_target_with_few_steps(self) -> None:
        initial = torch.zeros(2, 1, 4, 4)
        for solver in ("euler", "heun"):
            result = integrate_flow(lambda x, t: torch.ones_like(x) * 3, initial, 4, solver)
            self.assertTrue(torch.allclose(result, torch.full_like(initial, 3.0)))

    def test_heun_improves_linear_ode_accuracy(self) -> None:
        initial = torch.ones(1, 1, 2, 2)
        euler = integrate_flow(lambda x, t: x, initial, 4, "euler")
        heun = integrate_flow(lambda x, t: x, initial, 4, "heun")
        target = torch.exp(torch.ones_like(initial))
        self.assertLess((heun - target).abs().mean(), (euler - target).abs().mean())
        self.assertEqual(float(model_time(1.0, 1, torch.device("cpu"))[0]), 999.0)

    def test_guidance_descends_membrane_objective(self) -> None:
        class ZeroArchitect:
            def __call__(self, x, t):
                return type("Output", (), {"sample": torch.zeros_like(x)})()

        context = SamplingContext(
            architect=ZeroArchitect(), engineer=None,
            architect_stats={}, engineer_stats={},
            fz_condition=torch.zeros(1, 1, 2, 2),
            device=torch.device("cpu"), source_file=None,
        )
        config = {"sampling": {"guidance_scale": 1.0, "grad_clip": 5.0,
                               "guidance_schedule": "bell", "guide_w_max": 1.0,
                               "bell_peak": 0.5, "bell_width": 1.0}}
        with patch("tfm_shells.sampling.guided._mf_from_engineer", side_effect=lambda ctx, x, t: 0.5 + 0.1 * x.mean(dim=(1, 2, 3))):
            result, history = generate_samples(context, config, torch.zeros(1, 1, 2, 2), 4)
        self.assertGreater(float(result.mean()), 0.0)
        self.assertEqual(len(history["t"]), 4)

    def test_pbunet_training_interface_uses_flow_states(self) -> None:
        class TinyThreeBranch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.u = torch.nn.Conv2d(2, 1, 1)
                self.m = torch.nn.Conv2d(2, 6, 1)
                self.f = torch.nn.Conv2d(2, 6, 1)

            def forward(self, x, t):
                self.last_time = t.detach().clone()
                return type("Output", (), {"sample": torch.cat([self.u(x), self.m(x), self.f(x)], dim=1)})()

        model = TinyThreeBranch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = {
            "z": torch.zeros(2, 1, 4, 4), "fz_norm": torch.zeros(2, 1, 4, 4),
            "type_channel": torch.full((2, 1, 4, 4), -1.0),
            "fz_real": torch.zeros(2, 1, 4, 4), "physics": torch.zeros(2, 13, 4, 4),
            "ds": torch.ones(2, 1, 4, 4), "dv": torch.ones(2, 1, 4, 4),
            "mf_true": torch.zeros(2, 1, 4, 4),
        }
        stats = {"physics_mean": torch.zeros(13, 1, 1).tolist(),
                 "physics_std": torch.ones(13, 1, 1).tolist()}
        config = {"training": {"mixed_precision": False, "timestep_power": 2.0,
                               "warmup_epochs": 10, "epochs": 20, "grad_clip_norm": 1.0}}
        result = _run_epoch(model, [batch], optimizer, stats, config, torch.device("cpu"),
                            lambda_epoch=0.0, include_fz_channel=True,
                            include_type_channel=False, epoch=1, total_epochs=20, phase="test")
        self.assertGreaterEqual(result["mse"], 0.0)
        self.assertEqual(tuple(model.last_time.shape), (2,))
        self.assertTrue(torch.all(model.last_time >= 0) and torch.all(model.last_time <= 999))

    def test_architect_training_interface_uses_velocity_target(self) -> None:
        class TinyArchitect(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x, t):
                self.last_time = t.detach().clone()
                return type("Output", (), {"sample": self.layer(x)})()

        model = TinyArchitect()
        result = run_architect_epoch(
            model, [{"z": torch.zeros(2, 1, 4, 4)}],
            torch.optim.Adam(model.parameters(), lr=1e-3),
            torch.device("cpu"), False, 1.0, 1, 1, "test",
        )
        self.assertGreater(result["loss"], 0.0)
        self.assertTrue(torch.all(model.last_time >= 0) and torch.all(model.last_time <= 999))

    def test_real_unet_interfaces_accept_continuous_flow_time(self) -> None:
        base = {"sample_size": 8, "in_channels": 1, "out_channels": 1,
                "layers_per_block": 1, "block_out_channels": [32],
                "down_block_types": ["DownBlock2D"], "up_block_types": ["UpBlock2D"]}
        architect = build_unet(base)
        engineer = build_unet({**base, "kind": "parallel_pb_unet", "in_channels": 2,
                               "out_channels": 13, "branch_channels": {"u": 1, "m": 6, "f": 6}})
        time = model_time(0.375, 1, torch.device("cpu"))
        with torch.no_grad():
            self.assertEqual(tuple(architect(torch.randn(1, 1, 8, 8), time).sample.shape), (1, 1, 8, 8))
            self.assertEqual(tuple(engineer(torch.randn(1, 2, 8, 8), time).sample.shape), (1, 13, 8, 8))
        context = SamplingContext(
            architect, engineer,
            {"z_min": 0.0, "z_max": 1.0},
            {"z_min": 0.0, "z_max": 1.0,
             "physics_mean": torch.zeros(13, 1, 1).tolist(),
             "physics_std": torch.ones(13, 1, 1).tolist()},
            torch.zeros(1, 1, 8, 8), torch.device("cpu"), None,
        )
        config = {"sampling": {"guidance_scale": 0.1, "grad_clip": 5.0,
                               "guidance_schedule": "bell", "guide_w_max": 1.0,
                               "bell_peak": 0.5, "bell_width": 0.3}}
        states, _ = generate_samples(context, config, torch.randn(1, 1, 8, 8), 2)
        self.assertEqual(tuple(states.shape), (1, 1, 8, 8))
        self.assertTrue(torch.isfinite(states).all())


if __name__ == "__main__":
    unittest.main()
