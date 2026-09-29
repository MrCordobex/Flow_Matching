import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfm_shells.models.factory import build_unet
from tfm_shells.models.wavelet import starlet

CONFIG = {
    "kind": "wavelet_split", "backbone": "hybrid_fourier_unet",
    "wavelet": {"levels": 3, "threshold": 3.0, "learnable_threshold": True},
    "sample_size": 32, "in_channels": 2, "out_channels": 13,
    "base_channels": 8, "spectral_modes": 3, "spectral_layers": 1, "fft_padding": 2,
    "time_embedding_dim": 32, "branch_channels": {"u": 1, "m": 6, "f": 6},
}


class WaveletSplitTest(unittest.TestCase):
    def test_starlet_reconstructs_exactly(self) -> None:
        x = torch.randn(2, 1, 32, 32, dtype=torch.float64)
        details, coarse = starlet(x, 3)
        self.assertTrue(torch.allclose(coarse + sum(details), x))

    def test_split_adds_up_and_is_clean_at_t0(self) -> None:
        model = build_unet(CONFIG)
        heights = torch.randn(3, 1, 32, 32)
        for t in (0.0, 500.0, 999.0):
            shell, noise = model.split(heights, torch.full((3,), t))
            self.assertTrue(torch.allclose(shell + noise, heights, atol=1e-5))
        # t = 0 on the DDPM grid still has sigma ~ 0.006, so almost everything is shell.
        _, noise = model.split(heights, 0.0)
        self.assertLess(float(noise.square().mean()), 1e-3 * float(heights.square().mean()))

    def test_noise_channel_grows_with_noise_level(self) -> None:
        model = build_unet(CONFIG)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
        clean = (1.0 - x.square() - y.square()).view(1, 1, 32, 32)  # smooth dome
        energy = []
        for t in (100, 500, 900):
            alpha, sigma = model.alphas_cumprod[t].sqrt(), (1 - model.alphas_cumprod[t]).sqrt()
            state = alpha * clean + sigma * torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                _, noise = model.split(state, float(t))
            energy.append(float(noise.square().mean()))
        self.assertLess(energy[0], energy[1])
        self.assertLess(energy[1], energy[2])

    def test_forward_shape_and_gradients(self) -> None:
        model = build_unet(CONFIG)
        inputs = torch.randn(2, 2, 32, 32, requires_grad=True)
        prediction = model(inputs, torch.tensor([10.0, 700.0])).sample
        self.assertEqual(tuple(prediction.shape), (2, 13, 32, 32))
        prediction.square().mean().backward()
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.log_threshold.grad.abs().sum()), 0.0)

    def test_checkpoint_round_trip(self) -> None:
        model = build_unet(CONFIG).eval()
        clone = build_unet(CONFIG).eval()
        clone.load_state_dict(model.state_dict())
        inputs = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(inputs, 300.0).sample, clone(inputs, 300.0).sample))


BANDS = dict(CONFIG, wavelet={"mode": "bands", "levels": 3, "evidence": True})


class WaveletBandsTest(unittest.TestCase):
    def test_bands_add_up_and_evidence_is_bounded(self) -> None:
        model = build_unet(BANDS)
        heights = torch.randn(2, 1, 32, 32)
        front = model.bands(heights, torch.tensor([50.0, 800.0]))
        self.assertEqual(front.shape[1], 1 + 3 + 3)
        self.assertTrue(torch.allclose(front[:, :4].sum(dim=1, keepdim=True), heights, atol=1e-5))
        evidence = front[:, 4:]
        self.assertGreaterEqual(float(evidence.min()), 0.0)
        self.assertLessEqual(float(evidence.max()), 1.0)

    def test_evidence_falls_as_noise_grows(self) -> None:
        model = build_unet(BANDS)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
        clean = (0.5 * torch.cos(3 * x) * torch.cos(3 * y)).view(1, 1, 32, 32)
        noise = torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(0))
        means = []
        for t in (0, 300, 900):
            a, s = model.alphas_cumprod[t].sqrt(), (1 - model.alphas_cumprod[t]).sqrt()
            means.append(float(model.bands(a * clean + s * noise, float(t))[:, 4:].mean()))
        # Nearly clean: mostly trusted. The finest band of a smooth shape holds almost
        # no signal, so part of it sits at the (tiny) t = 0 noise level and scores 0.
        self.assertGreater(means[0], 0.7)
        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])

    def test_without_evidence_the_backbone_gets_bands_only(self) -> None:
        model = build_unet(dict(CONFIG, wavelet={"mode": "bands", "levels": 3, "evidence": False}))
        self.assertEqual(model.backbone.in_channels, 1 + 3 + 1)
        self.assertEqual(model.bands(torch.randn(1, 1, 32, 32), 100.0).shape[1], 4)

    def test_forward_shape_gradients_and_round_trip(self) -> None:
        model = build_unet(BANDS).eval()
        self.assertEqual(model.backbone.in_channels, 1 + 3 + 3 + 1)
        inputs = torch.randn(2, 2, 32, 32, requires_grad=True)
        prediction = model(inputs, torch.tensor([10.0, 700.0])).sample
        self.assertEqual(tuple(prediction.shape), (2, 13, 32, 32))
        prediction.square().mean().backward()
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        clone = build_unet(BANDS).eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(inputs, 300.0).sample, clone(inputs, 300.0).sample))

    def test_split_configs_without_mode_still_build_v1(self) -> None:
        model = build_unet(CONFIG)
        self.assertEqual(model.mode, "split")
        self.assertEqual(model.backbone.in_channels, 3)


if __name__ == "__main__":
    unittest.main()
