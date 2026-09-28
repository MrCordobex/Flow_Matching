"""Noise-aware wavelet split in front of a VP-conditioned surrogate.

The surrogate is queried on x_t = alpha_t * x0 + sigma_t * eps, and sigma_t is
known exactly from the timestep. An undecimated wavelet transform (a trous,
B3-spline "starlet") splits x_t into detail bands plus a coarse remainder that
add back to x_t exactly. White noise spreads over every detail coefficient
with a standard deviation that is computed here, whereas a smooth shell puts
its detail into few large coefficients (ridges, supports, curvature changes).

Each detail coefficient c of band j gets a trust weight

    w = c^2 / (c^2 + (k_j * sigma_t * s_j)^2)

where s_j is the band's noise level for unit white noise and k_j a learnable
multiplier: w -> 1 when c stands well above the noise, w -> 0 when c has the
size of the noise. Then

    shell = coarse + sum_j w_j * d_j,    noise = x_t - shell.

The two channels add up to x_t, so the split loses nothing: a surrogate fed
[shell, noise, fz] still targets E[R(x0) | x_t, t]. At t = 0 the noise channel
is zero and the backbone sees the clean shell.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

_B3_SPLINE = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)


def _smooth(x: torch.Tensor, dilation: int) -> torch.Tensor:
    """Separable B3-spline low-pass with holes of size `dilation` (reflect borders)."""
    kernel = torch.tensor(_B3_SPLINE, device=x.device, dtype=x.dtype)
    pad = 2 * dilation
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kernel.view(1, 1, 1, 5), dilation=(1, dilation))
    return F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), kernel.view(1, 1, 5, 1), dilation=(dilation, 1))


def starlet(x: torch.Tensor, levels: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Detail bands (fine to coarse) and coarse remainder; x == coarse + sum(details)."""
    details = []
    current = x
    for level in range(levels):
        smoother = _smooth(current, 2 ** level)
        details.append(current - smoother)
        current = smoother
    return details, current


class WaveletSplitSurrogate(nn.Module):
    """Splits the height channel into shell and noise parts, then calls the backbone.

    Input and output match the wrapped surrogate: (B, 1 + extra, H, W) with the
    noisy heights first, timestep on the 0..999 grid. The backbone receives one
    channel more, [shell, noise, extra...].
    """

    def __init__(
        self,
        backbone: nn.Module,
        sample_size: int,
        levels: int = 4,
        threshold: float = 3.0,
        learnable_threshold: bool = True,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
    ) -> None:
        super().__init__()
        if levels < 1 or 2 ** (levels + 1) >= sample_size:
            raise ValueError(f"levels must satisfy 1 <= levels and 2**(levels+1) < sample_size={sample_size}")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.backbone = backbone
        self.levels = levels

        # Noise std of each band for unit white noise: the L2 norm of the band's
        # impulse response, measured away from the borders.
        impulse = torch.zeros(1, 1, sample_size, sample_size, dtype=torch.float64)
        impulse[..., sample_size // 2, sample_size // 2] = 1.0
        details, _ = starlet(impulse, levels)
        noise_scale = torch.stack([band.square().sum().sqrt() for band in details]).float()
        self.register_buffer("noise_scale", noise_scale, persistent=False)

        # Same integer-grid cosine schedule as the Architect.
        from diffusers import DDPMScheduler

        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod.float(), persistent=False)

        log_k = torch.full((levels,), math.log(threshold))
        if learnable_threshold:
            self.log_threshold = nn.Parameter(log_k)
        else:
            self.register_buffer("log_threshold", log_k)

    @property
    def thresholds(self) -> torch.Tensor:
        """Current k_j per band, fine to coarse."""
        return self.log_threshold.detach().exp()

    def noise_sigma(self, timestep: torch.Tensor | float | int, batch: int, device: torch.device) -> torch.Tensor:
        t = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if t.numel() != batch:
            raise ValueError(f"Expected one timestep or {batch} timesteps, got {t.numel()}")
        index = t.round().long().clamp(0, self.alphas_cumprod.numel() - 1)
        return (1.0 - self.alphas_cumprod[index]).clamp_min(0.0).sqrt()

    def split(self, heights: torch.Tensor, timestep: torch.Tensor | float | int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (shell, noise) with shell + noise == heights."""
        x = heights.float()
        sigma = self.noise_sigma(timestep, x.shape[0], x.device).view(-1, 1, 1, 1)
        details, coarse = starlet(x, self.levels)
        k = self.log_threshold.exp()
        shell = coarse
        for level, band in enumerate(details):
            tau = k[level] * self.noise_scale[level] * sigma
            power = band.square()
            shell = shell + band * power / (power + tau.square() + 1e-12)
        return shell.to(heights.dtype), (x - shell).to(heights.dtype)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | float | int):
        shell, noise = self.split(sample[:, :1], timestep)
        return self.backbone(torch.cat((shell, noise, sample[:, 1:]), dim=1), timestep)
