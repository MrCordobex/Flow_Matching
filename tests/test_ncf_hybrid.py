import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfm_shells.models.factory import build_unet
from tfm_shells.models.ncf_hybrid import NCFBlock, gaussian_derivative_bank

PIECES = ("noise_condition", "calibrated_stem", "conditional_filters", "attention")


def config(size: int = 32, **ncf) -> dict:
    return {
        "kind": "ncf_hybrid", "sample_size": size, "in_channels": 2, "out_channels": 13,
        "base_channels": 8, "spectral_modes": 3, "spectral_layers": 1, "fft_padding": 2,
        "time_embedding_dim": 32, "branch_channels": {"u": 1, "m": 6, "f": 6},
        "ncf": {"filter_scales": [1.0, 2.0], "attention_heads": 2, **ncf},
    }


class NoiseCalibratedHybridTest(unittest.TestCase):
    def test_bank_is_zero_mean_except_smoothers(self) -> None:
        bank = gaussian_derivative_bank((1.0, 2.0)).flatten(1).sum(dim=1).view(2, 6)
        self.assertTrue(torch.allclose(bank[:, 0], torch.ones(2, dtype=bank.dtype)))
        self.assertLess(float(bank[:, 1:].abs().max()), 1e-10)

    def test_responses_carry_noise_of_known_std(self) -> None:
        model = build_unet(config(size=64))
        sigma = torch.tensor([0.5])
        noise = sigma * torch.randn(1, 1, 64, 64, generator=torch.Generator().manual_seed(0))
        response, _ = model.calibrated_responses(noise, sigma)
        std = response[..., 16:48, 16:48].std(dim=(0, 2, 3))  # interior, away from the reflect border
        # Each response is scaled to noise std sigma; the widest filters average few
        # independent samples in the window, hence the loose tolerance.
        self.assertTrue(torch.all((std / 0.5 - 1.0).abs() < 0.35), std)

    def test_z_is_bounded_even_when_nearly_clean(self) -> None:
        model = build_unet(config())
        _, alpha_sigma, _ = model.noise_level(0.0, 1, torch.device("cpu"))
        _, z = model.calibrated_responses(torch.rand(1, 1, 32, 32) * 2 - 1, alpha_sigma)
        self.assertTrue(torch.isfinite(z).all())
        self.assertLess(float(z.abs().max()), 20.0)

    def test_forward_shape_gradients_and_round_trip(self) -> None:
        model = build_unet(config()).eval()
        inputs = torch.randn(2, 2, 32, 32, requires_grad=True)
        prediction = model(inputs, torch.tensor([10.0, 700.0])).sample
        self.assertEqual(tuple(prediction.shape), (2, 13, 32, 32))
        prediction.square().mean().backward()
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        clone = build_unet(config()).eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(inputs, 300.0).sample, clone(inputs, 300.0).sample))
            early, late = model(inputs[:1], 0.0).sample, model(inputs[:1], 999.0).sample
        self.assertFalse(torch.allclose(early, late))

    def test_every_ablation_builds_and_runs(self) -> None:
        for piece in PIECES:
            model = build_unet(config(**{piece: False})).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.randn(1, 2, 32, 32), 500.0).sample.shape), (1, 13, 32, 32))
        model = build_unet(config(**{piece: False for piece in PIECES}))
        self.assertIsNone(model.attention)
        self.assertFalse(any(isinstance(m, NCFBlock) for m in model.modules()))

    def test_ncf_block_starts_as_identity(self) -> None:
        block = NCFBlock(8, 4, 16, dropout=0.0)
        x = torch.randn(2, 8, 16, 16)
        self.assertTrue(torch.equal(block(x, torch.randn(2, 4, 16, 16), torch.randn(2, 16)), x))


if __name__ == "__main__":
    unittest.main()
