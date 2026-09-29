"""Noise-aware wavelet front ends for VP-conditioned surrogates.

The surrogate is queried on x_t = alpha_t * x0 + sigma_t * eps, and sigma_t is
known exactly from the timestep. An undecimated wavelet transform (a trous,
B3-spline "starlet") splits x_t into detail bands plus a coarse remainder that
add back to x_t exactly. White noise spreads over every detail coefficient of
band j with a standard deviation sigma_t * s_j, where s_j is computed here,
whereas a smooth shell puts its detail into few large coefficients (ridges,
supports, curvature changes).

Two front ends, chosen with `mode`:

split  (v1) Each detail coefficient c gets a gate w = c^2 / (c^2 + (k_j sigma_t s_j)^2)
       with a learnable k_j, and the backbone receives
           shell = coarse + sum_j w_j d_j,    noise = x_t - shell.
       The gate cuts the signal: near SNR ~ 1 neighbouring coefficients land in
       different channels, and the backbone has to reassemble the shell.

bands  (v2) The backbone receives the signal intact, scale by scale,
           [coarse, d_1, ..., d_J]            (sums to x_t)
       and, with `evidence`, one confidence map per band,
           e_j = clamp(1 - (sigma_t s_j)^2 / P_j, 0, 1),
       where P_j is the local power of d_j (3x3 window at the band's scale).
       Since E[P_j] = alpha_t^2 * signal power + (sigma_t s_j)^2, e_j estimates
       the fraction of the local band power that is shell: 0 where the band does
       not stand out of the noise, 1 where it is almost all shell. This is the
       gain of the classical adaptive (local) Wiener filter, handed to the
       network as information instead of being applied to the signal. There are
       no thresholds to tune or learn: the noise level comes from the scheduler.

In both modes every channel is a function of (x_t, t) and the bands add back to
x_t, so nothing is lost and the surrogate still targets E[R(x0) | x_t, t]; the
front end only changes how the input is presented.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("split", "bands")
_B3_SPLINE = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)


def _smooth(x: torch.Tensor, dilation: int) -> torch.Tensor:
    """Separable B3-spline low-pass with holes of size `dilation` (reflect borders)."""
    kernel = torch.tensor(_B3_SPLINE, device=x.device, dtype=x.dtype)
    pad = 2 * dilation
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kernel.view(1, 1, 1, 5), dilation=(1, dilation))
    return F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), kernel.view(1, 1, 5, 1), dilation=(dilation, 1))


def _local_mean(x: torch.Tensor, dilation: int) -> torch.Tensor:
    """3x3 box average with holes of size `dilation`, matching the band's scale."""
    kernel = torch.full((1, 1, 3, 3), 1.0 / 9.0, device=x.device, dtype=x.dtype)
    return F.conv2d(F.pad(x, (dilation,) * 4, mode="reflect"), kernel, dilation=dilation)


def starlet(x: torch.Tensor, levels: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Detail bands (fine to coarse) and coarse remainder; x == coarse + sum(details)."""
    details = []
    current = x
    for level in range(levels):
        smoother = _smooth(current, 2 ** level)
        details.append(current - smoother)
        current = smoother
    return details, current


def extra_channels(levels: int, mode: str, evidence: bool) -> int:
    """How many channels the backbone receives beyond the wrapper's input."""
    if mode == "split":
        return 1                                 # heights -> shell, noise
    return levels + (levels if evidence else 0)  # heights -> coarse + J bands (+ J maps)


class WaveletSplitSurrogate(nn.Module):
    """Wavelet front end on the height channel, then the backbone.

    Input and output match the wrapped surrogate: (B, 1 + extra, H, W) with the
    noisy heights first, timestep on the 0..999 grid. The backbone receives
    `extra_channels(levels, mode, evidence)` channels more.
    """

    def __init__(
        self,
        backbone: nn.Module,
        sample_size: int,
        levels: int = 4,
        threshold: float = 3.0,
        learnable_threshold: bool = True,
        mode: str = "split",
        evidence: bool = True,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if levels < 1 or 2 ** (levels + 1) >= sample_size:
            raise ValueError(f"levels must satisfy 1 <= levels and 2**(levels+1) < sample_size={sample_size}")
        self.backbone = backbone
        self.levels = levels
        self.mode = mode
        self.evidence = evidence

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

        if mode == "split":
            if threshold <= 0:
                raise ValueError("threshold must be positive")
            log_k = torch.full((levels,), math.log(threshold))
            if learnable_threshold:
                self.log_threshold = nn.Parameter(log_k)
            else:
                self.register_buffer("log_threshold", log_k)

    @property
    def thresholds(self) -> torch.Tensor:
        """Current k_j per band, fine to coarse (split mode only)."""
        if self.mode != "split":
            raise AttributeError("bands mode has no thresholds")
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
        """Split mode: return (shell, noise) with shell + noise == heights."""
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

    def bands(self, heights: torch.Tensor, timestep: torch.Tensor | float | int) -> torch.Tensor:
        """Bands mode: [coarse, d_1..d_J] (sum == heights), then e_1..e_J if evidence."""
        x = heights.float()
        details, coarse = starlet(x, self.levels)
        channels = [coarse, *details]
        if self.evidence:
            sigma = self.noise_sigma(timestep, x.shape[0], x.device).view(-1, 1, 1, 1)
            for level, band in enumerate(details):
                noise_power = (self.noise_scale[level] * sigma).square()
                local_power = _local_mean(band.square(), 2 ** level)
                channels.append((1.0 - noise_power / (local_power + 1e-12)).clamp(0.0, 1.0))
        return torch.cat(channels, dim=1).to(heights.dtype)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | float | int):
        if self.mode == "split":
            shell, noise = self.split(sample[:, :1], timestep)
            front = torch.cat((shell, noise), dim=1)
        else:
            front = self.bands(sample[:, :1], timestep)
        return self.backbone(torch.cat((front, sample[:, 1:]), dim=1), timestep)
