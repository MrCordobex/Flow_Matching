"""The auxiliary VAE of the published paper, used only as a fixed metric space.

Reproduced here so the diversity numbers of the new sweep live in the same
latent the earlier convex-hull figures were drawn in. Encoder only: the
diversity metrics never decode.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 2) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean, which is what the paper's hull figures used."""
        hidden = self.encoder(x).flatten(1)
        return self.fc_mu(hidden)


def load_vae(checkpoint: dict[str, Any]) -> ConvVAE:
    config = checkpoint.get("model_config", {})
    model = ConvVAE(latent_dim=int(config.get("latent_dim", 2)))
    state = {key: value for key, value in checkpoint["model_state_dict"].items()
             if key.startswith(("encoder.", "fc_mu.", "fc_logvar."))}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
