"""Noise-calibrated conditional-filter hybrid (NCF-Hybrid) surrogate.

Same interface, trunk and heads as HybridFourierUNet: input [x_t, fz] and the
timestep on the 0..999 grid, output the 13 fields (u 1, m 6, f 6), trained on
R(x0) as before. Four pieces make the network aware of the known corruption
x_t = alpha_t x0 + sigma_t eps by construction, each switchable for ablation:

E1 noise_condition      alpha_t, sigma_t and log-SNR from the scheduler join the
                        sinusoidal embedding, and the resulting vector conditions
                        every block.
E2 calibrated_stem      A fixed bank of Gaussian-derivative filters (value, slope,
                        curvature) at several scales. For each filter h_k the
                        response r_k = h_k * x_t carries noise of std
                        sigma_t ||h_k|| exactly, so z_k = asinh(r_k / (sigma_t ||h_k||))
                        says how many noise standard deviations the response is.
                        Geometry and load get separate stems; z is also passed to
                        every block at its resolution.
E3 conditional_filters  LocalBlock -> NCFBlock: per-pixel scale/shift and a
                        per-pixel softmax mixture of K filters (1x1 .. 3x3 dil 4),
                        both computed from the features, z and the noise condition.
                        The block keeps F, the filtered S and the residual F - S.
E4 attention            One self-attention block at the 16x16 bottleneck.

No x0 estimate appears anywhere: every channel is a function of (x_t, t), so the
optimum is still E[R(x0) | x_t, t]. New layers start as the identity (zero-init
last projections), so training begins from a plain hybrid.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tfm_shells.models.hybrid_fourier_unet import (
    HybridBlock,
    HybridFourierOutput,
    LocalBlock,
    _time_embedding,
)

FILTERS_PER_SCALE = ("G", "Gx", "Gy", "Gxx", "Gyy", "Gxy")


def gaussian_derivative_bank(scales: tuple[float, ...]) -> torch.Tensor:
    """(6 * len(scales), 1, K, K) kernels: G and its first and second derivatives."""
    half = int(math.ceil(3.0 * max(scales)))
    grid = torch.arange(-half, half + 1, dtype=torch.float64)
    kernels = []
    for s in scales:
        g = torch.exp(-grid.square() / (2.0 * s * s))
        g = g / g.sum()
        g1 = -grid / (s * s) * g                          # odd: sums to zero exactly
        g2 = (grid.square() / s ** 4 - 1.0 / (s * s)) * g
        g2 = g2 - g * g2.sum()                            # discrete zero sum
        kernels += [torch.outer(g, g), torch.outer(g, g1), torch.outer(g1, g),
                    torch.outer(g, g2), torch.outer(g2, g), torch.outer(g1, g1)]
    return torch.stack(kernels).unsqueeze(1)


class NCFBlock(nn.Module):
    """Noise-calibrated conditional-filter block, a drop-in for LocalBlock."""

    def __init__(self, channels: int, z_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.context = nn.Conv2d(channels + z_channels, channels, 3, padding=1)
        self.context_time = nn.Linear(time_dim, channels)
        self.modulation = nn.Conv2d(channels, 2 * channels, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
        ])
        self.gate = nn.Conv2d(channels, len(self.branches), 1)
        self.mix = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1), nn.SiLU(), nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # Identity at initialisation: no modulation, uniform gate, zero update.
        for layer in (self.modulation, self.gate, self.mix[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None, cond: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        context_input = normed if z is None else torch.cat((normed, z), dim=1)
        context = F.silu(self.context(context_input) + self.context_time(F.silu(cond))[:, :, None, None])
        scale, shift = self.modulation(context).chunk(2, dim=1)
        modulated = F.silu(normed * (1.0 + scale) + shift)
        weights = torch.softmax(self.gate(context), dim=1)
        structure = sum(weights[:, k:k + 1] * branch(modulated) for k, branch in enumerate(self.branches))
        residual = modulated - structure
        return x + self.mix(torch.cat((modulated, structure, residual), dim=1))


class BottleneckAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(batch, channels, height, width)


class NoiseCalibratedHybrid(nn.Module):
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
        noise_condition: bool = True,
        calibrated_stem: bool = True,
        filter_scales: tuple[float, ...] = (1.0, 2.0, 4.0),
        conditional_filters: bool = True,
        attention: bool = True,
        attention_heads: int = 4,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
    ) -> None:
        super().__init__()
        branch_channels = branch_channels or {"u": 1, "m": 6, "f": 6}
        if (sum(branch_channels.values()) != out_channels or out_channels != 13
                or tuple(branch_channels[k] for k in ("u", "m", "f")) != (1, 6, 6)):
            raise ValueError("NoiseCalibratedHybrid requires branches u=1, m=6, f=6")
        if in_channels < 2:
            raise ValueError("NoiseCalibratedHybrid expects [x_t, load...] with at least 2 channels")
        if base_channels < 8 or base_channels % 8 or sample_size % 4:
            raise ValueError("base_channels must be a multiple of 8 and sample_size divisible by 4")
        if (4 * base_channels) % attention_heads:
            raise ValueError("4 * base_channels must be divisible by attention_heads")
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_embedding_dim
        self.noise_condition = noise_condition
        self.calibrated_stem = calibrated_stem
        c = base_channels

        from diffusers import DDPMScheduler

        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod.float(), persistent=False)

        # E1: the noise condition shared by every block.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim + (3 if noise_condition else 0), time_embedding_dim * 2), nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )

        # E2: noise-calibrated filter bank, then separate geometry and load stems.
        if calibrated_stem:
            bank = gaussian_derivative_bank(tuple(float(s) for s in filter_scales))
            if bank.shape[-1] // 2 >= sample_size:
                raise ValueError("filter_scales too large for sample_size")
            self.register_buffer("filter_bank", bank.float(), persistent=False)
            self.register_buffer("filter_norm", bank.flatten(1).norm(dim=1).float(), persistent=False)
            z_channels = bank.shape[0]
            geometry_in = 2 * z_channels
        else:
            z_channels = 0
            geometry_in = 1
        self.geometry_stem = nn.Sequential(nn.Conv2d(geometry_in, c, 3, padding=1), nn.SiLU(),
                                           nn.Conv2d(c, c, 3, padding=1))
        self.load_stem = nn.Sequential(nn.Conv2d(in_channels - 1, c, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(c, c, 3, padding=1))
        self.fuse = nn.Conv2d(2 * c + 2, c, 3, padding=1)

        # E3: conditional-filter blocks where the hybrid has LocalBlocks.
        def block(channels: int) -> nn.Module:
            if conditional_filters:
                return NCFBlock(channels, z_channels, time_embedding_dim, dropout)
            return LocalBlock(channels, time_embedding_dim, dropout)

        self.enc1 = block(c)
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.enc2 = block(2 * c)
        self.mid_spectral = HybridBlock(2 * c, time_embedding_dim, spectral_modes, fft_padding, dropout)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)
        self.bottleneck = nn.ModuleList([
            HybridBlock(4 * c, time_embedding_dim, spectral_modes, fft_padding, dropout)
            for _ in range(spectral_layers)
        ])
        # E4: non-local context at 16x16.
        self.attention = BottleneckAttention(4 * c, attention_heads) if attention else None
        self.up1 = nn.Conv2d(6 * c, 2 * c, 3, padding=1)
        self.dec1 = block(2 * c)
        self.up2 = nn.Conv2d(3 * c, c, 3, padding=1)
        self.dec2 = block(c)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                nn.SiLU(), nn.Conv2d(c, channels, 1))
            for name, channels in branch_channels.items()
        })

    def noise_level(self, timestep: torch.Tensor | float | int, batch: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """alpha_t, sigma_t and log-SNR on the integer grid, one per sample."""
        t = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if t.numel() != batch:
            raise ValueError(f"Expected one timestep or {batch} timesteps, got {t.numel()}")
        bar = self.alphas_cumprod[t.round().long().clamp(0, self.alphas_cumprod.numel() - 1)]
        log_snr = (bar.clamp_min(1e-12).log() - (1.0 - bar).clamp_min(1e-12).log()).clamp(-20.0, 20.0)
        return bar.sqrt(), (1.0 - bar).clamp_min(0.0).sqrt(), log_snr

    def calibrated_responses(self, heights: torch.Tensor,
                             sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter responses scaled to noise std sigma_t, and z in noise standard deviations."""
        pad = self.filter_bank.shape[-1] // 2
        response = F.conv2d(F.pad(heights.float(), (pad,) * 4, mode="reflect"), self.filter_bank)
        response = response / self.filter_norm.view(1, -1, 1, 1)
        z = torch.asinh(response / sigma.view(-1, 1, 1, 1).clamp_min(1e-6))
        return response.to(heights.dtype), z.to(heights.dtype)

    @staticmethod
    def _run(block: nn.Module, x: torch.Tensor, z: torch.Tensor | None, cond: torch.Tensor) -> torch.Tensor:
        return block(x, z, cond) if isinstance(block, NCFBlock) else block(x, cond)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | float | int) -> HybridFourierOutput:
        if sample.ndim != 4 or sample.shape[1] != self.in_channels:
            raise ValueError(f"Expected input (B,{self.in_channels},H,W), got {tuple(sample.shape)}")
        batch, _, height, width = sample.shape
        if height % 4 or width % 4:
            raise ValueError("Spatial dimensions must be divisible by 4")
        alpha, sigma, log_snr = self.noise_level(timestep, batch, sample.device)
        embedding = _time_embedding(timestep, batch, sample.device, self.time_dim)
        if self.noise_condition:
            embedding = torch.cat((embedding, torch.stack((alpha, sigma, log_snr / 10.0), dim=1)), dim=1)
        cond = self.time_mlp(embedding)

        heights, load = sample[:, :1], sample[:, 1:]
        if self.calibrated_stem:
            response, z64 = self.calibrated_responses(heights, sigma)
            geometry = self.geometry_stem(torch.cat((response, z64), dim=1))
            z32 = F.avg_pool2d(z64, 2)
        else:
            geometry = self.geometry_stem(heights)
            z64 = z32 = None
        y = torch.linspace(-1, 1, height, device=sample.device, dtype=sample.dtype)
        x = torch.linspace(-1, 1, width, device=sample.device, dtype=sample.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).expand(batch, -1, -1, -1)
        hidden = self.fuse(torch.cat((geometry, self.load_stem(load), coords), dim=1))

        skip1 = self._run(self.enc1, hidden, z64, cond)
        skip2 = self.mid_spectral(self._run(self.enc2, self.down1(skip1), z32, cond), cond)
        hidden = self.down2(skip2)
        for block in self.bottleneck:
            hidden = block(hidden, cond)
        if self.attention is not None:
            hidden = self.attention(hidden)
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self._run(self.dec1, self.up1(torch.cat((hidden, skip2), dim=1)), z32, cond)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self._run(self.dec2, self.up2(torch.cat((hidden, skip1), dim=1)), z64, cond)
        return HybridFourierOutput(torch.cat(tuple(self.heads[name](hidden) for name in ("u", "m", "f")), dim=1))
