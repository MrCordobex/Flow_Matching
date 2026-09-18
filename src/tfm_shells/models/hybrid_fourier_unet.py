"""Multiscale local/spectral surrogate for VP-conditioned solid shells."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HybridFourierOutput:
    sample: torch.Tensor


def _time_embedding(timestep: torch.Tensor | float | int, batch: int, device: torch.device, dim: int) -> torch.Tensor:
    t = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1:
        t = t.expand(batch)
    if t.numel() != batch:
        raise ValueError(f"Expected one timestep or {batch} timesteps, got {t.numel()}")
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1))
    phase = t[:, None] * freq[None, :]
    encoded = torch.cat((phase.sin(), phase.cos()), dim=1)
    return F.pad(encoded, (0, dim - encoded.shape[1]))


class SpectralConv2d(nn.Module):
    """Low-frequency Fourier mixing with a replicated edge halo.

    The FFT runs in float32 because CUDA half-precision FFT is restricted to
    power-of-two sizes. The halo reduces the artificial periodic jump at the
    boundaries of a clamped shell.
    """

    def __init__(self, channels: int, modes: int, padding: int) -> None:
        super().__init__()
        self.modes = modes
        self.padding = padding
        scale = 1.0 / channels
        self.positive = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.negative = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.padding
        padded = F.pad(x.float(), (p, p, p, p), mode="replicate") if p else x.float()
        spectrum = torch.fft.rfft2(padded, norm="ortho")
        out = torch.zeros_like(spectrum)
        modes_h = min(self.modes, padded.shape[-2] // 2)
        modes_w = min(self.modes, spectrum.shape[-1])
        positive = torch.view_as_complex(self.positive.contiguous())
        negative = torch.view_as_complex(self.negative.contiguous())
        out[:, :, :modes_h, :modes_w] = torch.einsum(
            "bihw,iohw->bohw", spectrum[:, :, :modes_h, :modes_w], positive[:, :, :modes_h, :modes_w]
        )
        out[:, :, -modes_h:, :modes_w] = torch.einsum(
            "bihw,iohw->bohw", spectrum[:, :, -modes_h:, :modes_w], negative[:, :, :modes_h, :modes_w]
        )
        result = torch.fft.irfft2(out, s=padded.shape[-2:], norm="ortho")
        return (result[:, :, p:-p, p:-p] if p else result).to(x.dtype)


class LocalBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels * 2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=1)
        y = self.norm2(y) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.conv2(self.dropout(F.silu(y)))


class HybridBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, modes: int, padding: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.spectral = SpectralConv2d(channels, modes, padding)
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels * 2)
        self.post = LocalBlock(channels, time_dim, dropout)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.norm(x))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=1)
        y = self.spectral(y) + self.local(y)
        x = x + F.silu(y * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None])
        return self.post(x, time)


class HybridFourierUNet(nn.Module):
    """Shared Fourier/U-Net trunk and separate uz, membrane and flexion heads."""

    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        spectral_modes: int = 8,
        spectral_layers: int = 2,
        fft_padding: int = 4,
        time_embedding_dim: int = 128,
        dropout: float = 0.05,
        branch_channels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        branch_channels = branch_channels or {"u": 1, "m": 6, "f": 6}
        if (sum(branch_channels.values()) != out_channels or out_channels != 13
                or tuple(branch_channels[k] for k in ("u", "m", "f")) != (1, 6, 6)):
            raise ValueError("HybridFourierUNet requires branches u=1, m=6, f=6")
        if base_channels < 8 or base_channels % 8 or sample_size % 4:
            raise ValueError("base_channels must be a multiple of 8 and sample_size divisible by 4")
        if spectral_modes < 1 or spectral_layers < 1 or fft_padding < 0 or time_embedding_dim < 2:
            raise ValueError("Invalid Fourier or time-embedding settings")
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        c = base_channels
        self.time_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2), nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        self.lift = nn.Conv2d(in_channels + 2, c, 3, padding=1)
        self.enc1 = LocalBlock(c, time_embedding_dim, dropout)
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.enc2 = LocalBlock(2 * c, time_embedding_dim, dropout)
        self.mid_spectral = HybridBlock(2 * c, time_embedding_dim, spectral_modes, fft_padding, dropout)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)
        self.bottleneck = nn.ModuleList([
            HybridBlock(4 * c, time_embedding_dim, spectral_modes, fft_padding, dropout)
            for _ in range(spectral_layers)
        ])
        self.up1 = nn.Conv2d(6 * c, 2 * c, 3, padding=1)
        self.dec1 = LocalBlock(2 * c, time_embedding_dim, dropout)
        self.up2 = nn.Conv2d(3 * c, c, 3, padding=1)
        self.dec2 = LocalBlock(c, time_embedding_dim, dropout)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                nn.SiLU(), nn.Conv2d(c, channels, 1))
            for name, channels in branch_channels.items()
        })

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | float | int) -> HybridFourierOutput:
        if sample.ndim != 4 or sample.shape[1] != self.in_channels:
            raise ValueError(f"Expected input (B,{self.in_channels},H,W), got {tuple(sample.shape)}")
        batch, _, height, width = sample.shape
        if height % 4 or width % 4:
            raise ValueError("Spatial dimensions must be divisible by 4")
        y = torch.linspace(-1, 1, height, device=sample.device, dtype=sample.dtype)
        x = torch.linspace(-1, 1, width, device=sample.device, dtype=sample.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).expand(batch, -1, -1, -1)
        time = self.time_mlp(_time_embedding(timestep, batch, sample.device, self.time_dim))
        skip1 = self.enc1(self.lift(torch.cat((sample, coords), dim=1)), time)
        skip2 = self.mid_spectral(self.enc2(self.down1(skip1), time), time)
        hidden = self.down2(skip2)
        for block in self.bottleneck:
            hidden = block(hidden, time)
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec1(self.up1(torch.cat((hidden, skip2), dim=1)), time)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec2(self.up2(torch.cat((hidden, skip1), dim=1)), time)
        return HybridFourierOutput(torch.cat(tuple(self.heads[name](hidden) for name in ("u", "m", "f")), dim=1))
