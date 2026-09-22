from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from tfm_shells.cosine_flow import (
    FLOW_METHOD,
    CosineFlowPath,
    denoised_from_velocity,
    integrate_cosine_flow,
    project_velocity,
    sample_cosine_flow_path,
)
from tfm_shells.flow import integrate_flow, model_time
from tfm_shells.models.factory import build_unet
from tfm_shells.sampling.guided import SamplingContext, _check_vp_checkpoint, generate_samples
from tfm_shells.training.train_architect import _run_epoch as run_architect_epoch
from tfm_shells.training.train_engineer import _run_epoch
from tfm_shells.vp_diffusion import CosineVPSchedule, ddim_step, sample_ddim, sample_ddpm, sample_vp_path, vp_model_time
from tfm_shells.utils.physics import (
    balanced_gradient_loss,
    balanced_supervised_loss,
    branchwise_supervised_losses,
    compute_constitutive_loss,
)


class FlowMatchingTests(unittest.TestCase):
    def test_architect_engineer_configs_and_checkpoint_methods_match(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "architect.yaml").open(encoding="utf-8") as handle:
            architect = yaml.safe_load(handle)
        with (root / "configs" / "engineer.yaml").open(encoding="utf-8") as handle:
            engineer = yaml.safe_load(handle)
        with (root / "configs" / "sample_guided.yaml").open(encoding="utf-8") as handle:
            sampling = yaml.safe_load(handle)
        self.assertEqual(float(architect["model"]["cosine_s"]), float(engineer["model"]["cosine_s"]))
        self.assertEqual(sampling["sampling"]["solver"], "ddim")
        self.assertEqual(float(sampling["sampling"]["eta"]), 0.0)
        self.assertEqual(
            _check_vp_checkpoint({"diffusion_config": {"method": "cosine_vp_v", "time_scale": 999.0, "cosine_s": 0.008}},
                                 "Architect", "cosine_vp_v")["method"],
            "cosine_vp_v",
        )
        with self.assertRaisesRegex(ValueError, "retrain older flow checkpoints"):
            _check_vp_checkpoint({"flow_config": {"path": "linear", "time_scale": 999.0}},
                                 "Architect", "cosine_vp_v")

    def test_cosine_vp_path_and_ddim_recover_clean_image(self) -> None:
        schedule = CosineVPSchedule(0.008)
        clean = torch.full((2, 1, 4, 4), 0.7)
        noise = torch.randn_like(clean)
        alpha, sigma = schedule.alpha_sigma(torch.tensor(1.0))
        self.assertEqual(float(alpha), 0.0)
        self.assertAlmostEqual(float(alpha.square() + sigma.square()), 1.0, places=6)
        initial = sigma * noise
        def oracle(state: torch.Tensor, t: float) -> torch.Tensor:
            a, s = schedule.alpha_sigma(torch.tensor(t))
            return a * noise - s * clean
        for spacing in ("uniform", "quadratic"):
            sampled = sample_ddim(oracle, initial, 7, schedule, spacing=spacing, eta=0.0)
            self.assertTrue(torch.allclose(sampled, clean, atol=2e-6))

    def test_stochastic_ddim_step_uses_supplied_noise(self) -> None:
        schedule = CosineVPSchedule()
        state = torch.zeros(1, 1, 2, 2)
        velocity = torch.ones_like(state)
        noise = torch.ones_like(state)
        deterministic = ddim_step(state, velocity, 0.8, 0.4, schedule, eta=0.0)
        stochastic = ddim_step(state, velocity, 0.8, 0.4, schedule, eta=1.0, noise=noise)
        self.assertTrue(torch.isfinite(stochastic).all())
        self.assertFalse(torch.allclose(deterministic, stochastic))

    def test_ddpm_step_matches_vp_posterior_mean_and_variance(self) -> None:
        schedule = CosineVPSchedule()
        state = torch.tensor([[[[0.4, -0.2]]]], dtype=torch.float64)
        velocity = torch.tensor([[[[-0.5, 0.3]]]], dtype=torch.float64)
        noise = torch.tensor([[[[0.7, -1.2]]]], dtype=torch.float64)
        t, next_t = 0.8, 0.55
        alpha_t, sigma_t = schedule.alpha_sigma(torch.tensor(t, dtype=state.dtype))
        alpha_s, sigma_s = schedule.alpha_sigma(torch.tensor(next_t, dtype=state.dtype))
        x0 = alpha_t * state - sigma_t * velocity
        beta = 1 - (alpha_t / alpha_s).square()
        variance = beta * sigma_s.square() / sigma_t.square()
        mean = ((alpha_s * beta / sigma_t.square()) * x0
                + (alpha_t * sigma_s.square() / (alpha_s * sigma_t.square())) * state)
        sampled = ddim_step(state, velocity, t, next_t, schedule, eta=1.0, noise=noise)
        self.assertTrue(torch.allclose(sampled, mean + variance.sqrt() * noise, atol=1e-12))

    def test_ddpm_seed_and_x0_clipping_are_configurable(self) -> None:
        schedule = CosineVPSchedule()
        initial = torch.randn(2, 1, 4, 4)
        field = lambda x, t: torch.zeros_like(x)
        torch.manual_seed(13)
        a = sample_ddpm(field, initial.clone(), 4, schedule, clip_denoised=True)
        torch.manual_seed(13)
        b = sample_ddpm(field, initial.clone(), 4, schedule, clip_denoised=True)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.all(a.abs() <= 1))
        unclipped = sample_ddim(field, initial.clone(), 4, schedule, eta=0.0)
        self.assertFalse(torch.allclose(a, unclipped))

    def test_sampler_selection_works_with_guidance_disabled(self) -> None:
        class ZeroArchitect:
            def __call__(self, x, t):
                return type("Output", (), {"sample": torch.zeros_like(x)})()

        context = SamplingContext(
            architect=ZeroArchitect(), engineer=None, architect_stats={}, engineer_stats={},
            fz_condition=torch.zeros(1, 1, 4, 4), device=torch.device("cpu"), source_file=None,
        )
        config = {"sampling": {"guidance_scale": 0.0, "grad_clip": 5.0,
                               "eta": 0.0, "time_spacing": "uniform"}}
        initial = torch.randn(1, 1, 4, 4)
        deterministic, _ = generate_samples(context, config, initial.clone(), 4, "ddim")
        torch.manual_seed(19)
        ancestral, _ = generate_samples(context, config, initial.clone(), 4, "ddpm")
        self.assertTrue(torch.isfinite(ancestral).all())
        self.assertFalse(torch.allclose(deterministic, ancestral))

    def test_architect_and_engineer_share_vp_path(self) -> None:
        schedule = CosineVPSchedule(0.008)
        clean = torch.randn(8, 1, 4, 4)
        state, time, target = sample_vp_path(clean, schedule)
        alpha, sigma = schedule.alpha_sigma(time)
        a, s = alpha[:, None, None, None], sigma[:, None, None, None]
        noise = (state - a * clean) / s
        self.assertTrue(torch.allclose(target, a * noise - s * clean, atol=1e-5))
        self.assertEqual(float(vp_model_time(1.0, 1, torch.device("cpu"))[0]), 999.0)

    def test_cosine_flow_velocity_is_the_path_time_derivative(self) -> None:
        path = CosineFlowPath(0.008)
        clean = torch.randn(4, 1, 4, 4, dtype=torch.float64)
        noise = torch.randn_like(clean)
        t = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)

        def state_at(time: torch.Tensor) -> torch.Tensor:
            alpha, sigma = path.alpha_sigma(time)
            return alpha * clean + sigma * noise

        # Autograd through the path must reproduce the analytic flow velocity.
        numeric = torch.autograd.grad(state_at(t).sum(), t)[0]
        alpha, sigma = path.alpha_sigma(t.detach())
        analytic = path.velocity_from_v(alpha * noise - sigma * clean)
        self.assertTrue(torch.allclose(numeric, analytic.sum(), atol=1e-9))
        self.assertAlmostEqual(path.rate, torch.pi / (2 * 1.008), places=12)

    def test_flow_target_is_the_v_target_scaled_by_the_angular_rate(self) -> None:
        path = CosineFlowPath(0.008)
        clean = torch.randn(8, 1, 4, 4)
        torch.manual_seed(5)
        vp_state, vp_time, vp_target = sample_vp_path(clean, path.schedule)
        torch.manual_seed(5)
        flow_state, flow_time, flow_target = sample_cosine_flow_path(clean, path)
        # Identical states and times keep the Engineer and its 999t embedding valid.
        self.assertTrue(torch.equal(vp_state, flow_state))
        self.assertTrue(torch.equal(vp_time, flow_time))
        self.assertTrue(torch.allclose(flow_target, path.rate * vp_target, atol=1e-6))
        self.assertTrue(torch.allclose(path.v_from_velocity(flow_target), vp_target, atol=1e-6))

    def test_exponential_solver_reproduces_ddim_on_the_same_field(self) -> None:
        path = CosineFlowPath(0.008)
        initial = torch.randn(2, 1, 4, 4)
        constant = torch.randn(2, 1, 4, 4)
        ddim = sample_ddim(lambda x, t: constant, initial.clone(), 6, path.schedule,
                           spacing="quadratic", eta=0.0)
        flow = integrate_cosine_flow(lambda x, t: path.velocity_from_v(constant),
                                     initial.clone(), 6, path,
                                     spacing="quadratic", solver="exponential")
        self.assertTrue(torch.allclose(ddim, flow, atol=1e-6))

    @staticmethod
    def _oracle_problem(seed: int = 0):
        """Exact cosine path plus the analytic field that generates it."""
        path = CosineFlowPath(0.008)
        generator = torch.Generator().manual_seed(seed)
        clean = torch.full((1, 1, 4, 4), 0.35, dtype=torch.float64)
        noise = torch.randn(1, 1, 4, 4, dtype=torch.float64, generator=generator)
        alpha, sigma = path.alpha_sigma(torch.tensor(1.0, dtype=torch.float64))
        initial = alpha * clean + sigma * noise

        def oracle(state: torch.Tensor, t: float) -> torch.Tensor:
            a, s = path.alpha_sigma(torch.tensor(t, dtype=state.dtype))
            return path.velocity_from_v(a * noise - s * clean)

        return path, clean, noise, initial, oracle

    def test_each_solver_converges_at_its_theoretical_order(self) -> None:
        path, clean, _, initial, oracle = self._oracle_problem()
        for solver, expected in (("euler", 1.0), ("midpoint", 2.0), ("heun", 2.0), ("rk4", 4.0)):
            with self.subTest(solver=solver):
                errors = []
                for steps in (32, 64, 128):
                    result = integrate_cosine_flow(oracle, initial.clone(), steps, path,
                                                   spacing="uniform", solver=solver)
                    errors.append(float((result - clean).abs().max()))
                # Halving the step must divide the error by 2**order.
                for coarse, fine in zip(errors, errors[1:]):
                    order = math.log2(coarse / fine)
                    self.assertAlmostEqual(order, expected, delta=0.15)

    def test_exponential_solver_is_exact_for_the_cosine_path(self) -> None:
        path, clean, _, initial, oracle = self._oracle_problem()
        for steps in (2, 8, 64):
            result = integrate_cosine_flow(oracle, initial.clone(), steps, path,
                                           spacing="uniform", solver="exponential")
            self.assertLess(float((result - clean).abs().max()), 1e-12)

    def test_final_denoise_removes_the_residual_noise_left_by_the_cosine_offset(self) -> None:
        path, clean, noise, initial, oracle = self._oracle_problem()
        alpha_zero, sigma_zero = path.alpha_sigma(torch.tensor(0.0, dtype=torch.float64))
        self.assertGreater(float(sigma_zero), 0.01)

        raw = integrate_cosine_flow(oracle, initial.clone(), 256, path, spacing="uniform",
                                    solver="rk4", final_denoise=False)
        denoised = integrate_cosine_flow(oracle, initial.clone(), 256, path, spacing="uniform",
                                         solver="rk4", final_denoise=True)
        # Without the closing projection the ODE lands on the true t=0 state,
        # which still carries sigma(0) * eps of white noise per pixel.
        self.assertTrue(torch.allclose(raw, alpha_zero * clean + sigma_zero * noise, atol=1e-8))
        self.assertLess(float((denoised - clean).abs().max()), 1e-8)
        self.assertGreater(float((raw - clean).abs().max()), 0.01)

    def test_velocity_projection_is_identity_inside_the_range(self) -> None:
        path = CosineFlowPath(0.008)
        state = torch.randn(2, 1, 4, 4, dtype=torch.float64)
        t = 0.6
        alpha, sigma = path.alpha_sigma(torch.tensor(t, dtype=torch.float64))
        clean = torch.full_like(state, 0.5)
        noise = (state - alpha * clean) / sigma
        velocity = path.velocity_from_v(alpha * noise - sigma * clean)
        self.assertTrue(torch.allclose(project_velocity(state, velocity, t, path),
                                       velocity, atol=1e-9))
        recovered, _ = denoised_from_velocity(state, velocity, t, path)
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-9))

    def test_velocity_projection_clamps_an_out_of_range_shell(self) -> None:
        path = CosineFlowPath(0.008)
        state = torch.randn(2, 1, 4, 4, dtype=torch.float64)
        t = 0.6
        alpha, sigma = path.alpha_sigma(torch.tensor(t, dtype=torch.float64))
        clean = torch.full_like(state, 4.0)
        noise = (state - alpha * clean) / sigma
        velocity = path.velocity_from_v(alpha * noise - sigma * clean)
        projected = project_velocity(state, velocity, t, path)
        self.assertFalse(torch.allclose(projected, velocity))
        # Projection is idempotent: the rebuilt clean estimate is exactly the clamp.
        reclean, _ = denoised_from_velocity(state, projected, t, path)
        self.assertTrue(torch.allclose(reclean, torch.ones_like(clean), atol=1e-9))
        self.assertTrue(torch.allclose(project_velocity(state, projected, t, path),
                                       projected, atol=1e-9))

    def test_flow_checkpoint_is_rescaled_to_v_units_when_sampling(self) -> None:
        path = CosineFlowPath(0.008)
        seen: list[torch.Tensor] = []

        class FlowArchitect:
            def __call__(self, x, t):
                velocity = torch.full_like(x, 0.25)
                seen.append(velocity)
                return type("Output", (), {"sample": velocity})()

        context = SamplingContext(
            architect=FlowArchitect(), engineer=None, architect_stats={}, engineer_stats={},
            fz_condition=torch.zeros(1, 1, 4, 4), device=torch.device("cpu"), source_file=None,
            vp_schedule=path.schedule, flow_path=path, net_to_v=1.0 / path.rate,
            architect_method=FLOW_METHOD,
        )
        config = {"sampling": {"guidance_scale": 0.0, "grad_clip": 5.0,
                               "eta": 0.0, "time_spacing": "uniform"}}
        initial = torch.randn(1, 1, 4, 4)
        # A dx/dt network driven through DDIM must equal a dx/dphi network
        # carrying the same field divided by the angular rate.
        flow_result, _ = generate_samples(context, config, initial.clone(), 5, "ddim")
        reference = sample_ddim(lambda x, t: torch.full_like(x, 0.25 / path.rate),
                                initial.clone(), 5, path.schedule, spacing="uniform", eta=0.0)
        self.assertTrue(torch.allclose(flow_result, reference, atol=1e-6))

    def test_guided_sampling_runs_under_every_ode_solver(self) -> None:
        path = CosineFlowPath(0.008)

        class ZeroArchitect:
            def __call__(self, x, t):
                return type("Output", (), {"sample": torch.zeros_like(x)})()

        context = SamplingContext(
            architect=ZeroArchitect(), engineer=None, architect_stats={}, engineer_stats={},
            fz_condition=torch.zeros(1, 1, 4, 4), device=torch.device("cpu"), source_file=None,
            vp_schedule=path.schedule, flow_path=path, net_to_v=1.0 / path.rate,
            architect_method=FLOW_METHOD,
        )
        config = {"sampling": {"guidance_scale": 0.0, "grad_clip": 5.0, "eta": 0.0,
                               "time_spacing": "quadratic", "clip_denoised": True}}
        initial = torch.randn(1, 1, 4, 4)
        for solver in ("exponential", "euler", "midpoint", "heun", "rk4"):
            with self.subTest(solver=solver):
                states, history = generate_samples(context, config, initial.clone(), 4, solver)
                self.assertTrue(torch.isfinite(states).all())
                self.assertLessEqual(float(states.abs().max()), 1.0 + 1e-6)
                self.assertEqual(len(history["t"]), 4)
        with self.assertRaisesRegex(ValueError, "solver must be"):
            generate_samples(context, config, initial.clone(), 4, "dopri5")

    def test_architect_training_accepts_the_flow_method(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "architect.yaml").open(encoding="utf-8") as handle:
            architect = yaml.safe_load(handle)
        self.assertEqual(architect["model"]["method"], FLOW_METHOD)

        path = CosineFlowPath(0.008)

        class ConstantArchitect(torch.nn.Module):
            """Emits a fixed field, scaled to whichever parameterization is used."""

            def __init__(self, scale: float):
                super().__init__()
                self.scale = scale

            def forward(self, x, t):
                return type("Output", (), {"sample": torch.full_like(x, 0.3 * self.scale)})()

        batch = [{"z": torch.randn(4, 1, 4, 4)}]
        torch.manual_seed(3)
        flow = run_architect_epoch(ConstantArchitect(path.rate), batch, path.schedule, None,
                                   torch.device("cpu"), False, 1.0, 1, 1, "test", flow_path=path)
        torch.manual_seed(3)
        vp = run_architect_epoch(ConstantArchitect(1.0), batch, path.schedule, None,
                                 torch.device("cpu"), False, 1.0, 1, 1, "test")
        # Equivalent predictors on identical states: the flow objective is exactly
        # k^2 times the v objective, and loss_v_units undoes that so a flow run can
        # be read against the curves of a v run.
        self.assertAlmostEqual(flow["loss"], vp["loss"] * path.rate ** 2, places=5)
        self.assertAlmostEqual(flow["loss_v_units"], vp["loss"], places=5)
        self.assertAlmostEqual(vp["loss_v_units"], vp["loss"], places=9)

    def test_balanced_supervision_weights_each_branch_equally(self) -> None:
        pred = torch.zeros(1, 13, 2, 2)
        pred[:, 0] = 1.0
        target = torch.zeros_like(pred)
        losses = branchwise_supervised_losses(pred, target)
        self.assertAlmostEqual(float(balanced_supervised_loss(losses)), 1.0 / 3.0)
        self.assertAlmostEqual(float((pred - target).square().mean()), 1.0 / 13.0)

    def test_gradient_loss_detects_spatial_errors_in_each_branch(self) -> None:
        target = torch.zeros(1, 13, 8, 8)
        for channel in (0, 1, 7):
            pred = target.clone()
            pred[:, channel, 3:5, 3:5] = 1.0
            self.assertGreater(float(balanced_gradient_loss(pred, target)), 0.0)
        self.assertEqual(float(balanced_gradient_loss(target, target)), 0.0)

    def test_hybrid_fourier_surrogate_predicts_13_fields_and_backpropagates(self) -> None:
        model = build_unet({
            "kind": "hybrid_fourier_unet", "sample_size": 16, "in_channels": 2,
            "out_channels": 13, "base_channels": 8, "spectral_modes": 3,
            "spectral_layers": 1, "fft_padding": 2, "time_embedding_dim": 32,
            "branch_channels": {"u": 1, "m": 6, "f": 6},
        })
        inputs = torch.randn(2, 2, 16, 16, requires_grad=True)
        prediction = model(inputs, torch.tensor([0.0, 999.0])).sample
        self.assertEqual(tuple(prediction.shape), (2, 13, 16, 16))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.bottleneck[0].spectral.positive.grad.abs().sum()), 0.0)
        with torch.no_grad():
            early = model(inputs[:1], torch.tensor([0.0])).sample
            late = model(inputs[:1], torch.tensor([999.0])).sample
        self.assertFalse(torch.allclose(early, late))

    def test_hybrid_fourier_surrogate_trains_with_gradient_loss_on_vp_states(self) -> None:
        model = build_unet({
            "kind": "hybrid_fourier_unet", "sample_size": 8, "in_channels": 2,
            "out_channels": 13, "base_channels": 8, "spectral_modes": 2,
            "spectral_layers": 1, "fft_padding": 1, "time_embedding_dim": 32,
        })
        batch = {
            "z": torch.randn(1, 1, 8, 8), "fz_norm": torch.zeros(1, 1, 8, 8),
            "type_channel": torch.full((1, 1, 8, 8), -1.0),
            "fz_real": torch.zeros(1, 1, 8, 8), "physics": torch.randn(1, 13, 8, 8),
            "ds": torch.ones(1, 1, 8, 8), "dv": torch.ones(1, 1, 8, 8),
            "mf_true": torch.zeros(1, 1, 8, 8),
        }
        stats = {"physics_mean": torch.zeros(13, 1, 1).tolist(),
                 "physics_std": torch.ones(13, 1, 1).tolist()}
        config = {"training": {"mixed_precision": False, "timestep_power": 2.0,
                               "warmup_epochs": 0, "epochs": 2, "grad_clip_norm": 1.0,
                               "gradient": {"enabled": True, "lambda_max": 0.05,
                                            "warmup_epochs": 0}}}
        result = _run_epoch(
            model, [batch], CosineVPSchedule(), torch.optim.Adam(model.parameters(), lr=1e-3),
            stats, config, torch.device("cpu"), lambda_epoch=0.0,
            include_fz_channel=True, include_type_channel=False,
            epoch=2, total_epochs=2, phase="test",
        )
        self.assertGreater(result["gradient"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_constitutive_loss_matches_exported_shell_convention(self) -> None:
        std = torch.ones(1, 13, 1, 1)
        std[:, 1:4] = 1e-5
        std[:, 4:7] = 1e4
        std[:, 7:10] = 1e-4
        std[:, 10:13] = 1e3
        mean = torch.zeros_like(std)
        real = torch.zeros(1, 13, 2, 2)
        real[:, 1:4] = torch.tensor([1e-5, 2e-5, 3e-5]).view(1, 3, 1, 1)
        real[:, 7:10] = torch.tensor([1e-4, 2e-4, 3e-4]).view(1, 3, 1, 1)
        a = 30e9 * 0.1 / (1 - 0.2**2)
        d = 30e9 * 0.1**3 / (12 * (1 - 0.2**2))
        real[:, 4:7] = torch.tensor([
            a * (1e-5 + 0.2 * 2e-5),
            a * (2e-5 + 0.2 * 1e-5),
            a * 0.8 * 0.5 * 3e-5,
        ]).view(1, 3, 1, 1)
        real[:, 10:13] = torch.tensor([
            d * (1e-4 + 0.2 * 2e-4),
            d * (2e-4 + 0.2 * 1e-4),
            d * 0.8 * 0.5 * 3e-4,
        ]).view(1, 3, 1, 1)
        exact = (real / std).detach().requires_grad_()
        loss = compute_constitutive_loss(exact, mean, std, 30e9, 0.2, 0.1)
        self.assertLess(float(loss.detach()), 1e-11)
        inconsistent = exact.detach().clone()
        inconsistent[:, 4] += 1.0
        inconsistent.requires_grad_()
        loss = compute_constitutive_loss(inconsistent, mean, std, 30e9, 0.2, 0.1)
        self.assertGreater(float(loss.detach()), 0.1)
        loss.backward()
        self.assertGreater(float(inconsistent.grad[:, 4].abs().sum().detach()), 0.0)

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

    def test_pbunet_training_interface_uses_vp_states(self) -> None:
        class TinyThreeBranch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.u = torch.nn.Conv2d(2, 1, 1)
                self.m = torch.nn.Conv2d(2, 6, 1)
                self.f = torch.nn.Conv2d(2, 6, 1)

            def forward(self, x, t):
                if not hasattr(self, "calls"):
                    self.calls = []
                self.calls.append((x.detach().clone(), t.detach().clone()))
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
                               "warmup_epochs": 10, "epochs": 20, "grad_clip_norm": 1.0,
                               "constitutive": {"enabled": True, "young_modulus": 1.0,
                                                "poisson_ratio": 0.2, "thickness": 1.0,
                                                "lambda_max": 0.05, "warmup_epochs": 0}}}
        result = _run_epoch(model, [batch], CosineVPSchedule(), optimizer, stats, config, torch.device("cpu"),
                            lambda_epoch=0.0, include_fz_channel=True,
                            include_type_channel=False, epoch=1, total_epochs=20, phase="test")
        self.assertGreaterEqual(result["mse"], 0.0)
        self.assertGreaterEqual(result["constitutive"], 0.0)
        self.assertEqual(tuple(model.last_time.shape), (2,))
        self.assertTrue(torch.all(model.last_time == 0))
        self.assertTrue(torch.all(model.calls[0][1] >= 0) and torch.all(model.calls[0][1] <= 999))

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
            model, [{"z": torch.zeros(2, 1, 4, 4)}], CosineVPSchedule(),
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
        engineer = build_unet({
            "kind": "hybrid_fourier_unet", "sample_size": 8, "in_channels": 2,
            "out_channels": 13, "base_channels": 8, "spectral_modes": 2,
            "spectral_layers": 1, "fft_padding": 1, "time_embedding_dim": 32,
            "branch_channels": {"u": 1, "m": 6, "f": 6},
        })
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
