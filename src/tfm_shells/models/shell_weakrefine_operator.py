from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShellWeakRefineOperatorOutput:
    sample: torch.Tensor
    log_variance: torch.Tensor | None = None


def _timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.unsqueeze(1)

    exponent = -math.log(float(max_period)) * torch.arange(
        half_dim,
        device=timesteps.device,
        dtype=torch.float32,
    ) / max(half_dim, 1)
    frequencies = torch.exp(exponent)
    angles = timesteps.unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class SpectralOperator2d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_height: int,
        modes_width: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.modes_height = modes_height
        self.modes_width = modes_width

        scale = 1.0 / max(channels, 1)
        self.weight_top = nn.Parameter(scale * torch.randn(channels, channels, modes_height, modes_width, 2))
        self.weight_bottom = nn.Parameter(scale * torch.randn(channels, channels, modes_height, modes_width, 2))

    @staticmethod
    def _complex_mul(x_ft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x_ft, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mh = min(self.modes_height, height)
        mw = min(self.modes_width, width // 2 + 1)
        weight_top = torch.view_as_complex(self.weight_top[:, :, :mh, :mw].contiguous())
        weight_bottom = torch.view_as_complex(self.weight_bottom[:, :, :mh, :mw].contiguous())

        out_ft[:, :, :mh, :mw] = self._complex_mul(x_ft[:, :, :mh, :mw], weight_top)
        out_ft[:, :, -mh:, :mw] = self._complex_mul(x_ft[:, :, -mh:, :mw], weight_bottom)
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class SharedOperatorBlock(nn.Module):
    def __init__(
        self,
        width: int,
        modes_height: int,
        modes_width: int,
        time_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(1, width)
        self.norm_2 = nn.GroupNorm(1, width)
        self.spectral = SpectralOperator2d(width, modes_height, modes_width)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.time_proj = nn.Linear(time_embedding_dim, width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_bias = self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)

        residual = x
        x = self.norm_1(x)
        x = self.spectral(x) + self.pointwise(x) + time_bias
        x = F.gelu(x)
        x = x + residual

        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = F.gelu(x)
        return x + residual


class BranchDecoder(nn.Module):
    def __init__(
        self,
        width: int,
        out_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(width, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ShellWeakRefineOperator(nn.Module):
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        operator_width: int = 128,
        num_operator_layers: int = 6,
        spectral_modes_height: int = 16,
        spectral_modes_width: int = 16,
        time_embedding_dim: int = 256,
        branch_hidden_channels: int = 128,
        branch_channels: dict[str, int] | None = None,
        use_coordinate_grid: bool = True,
        predict_log_variance: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_coordinate_grid = use_coordinate_grid
        self.predict_log_variance = predict_log_variance
        self.time_embedding_dim = time_embedding_dim

        branch_channels = branch_channels or {"u": 1, "m": 6, "f": 6}
        total_branch_channels = sum(int(value) for value in branch_channels.values())
        if total_branch_channels != out_channels:
            raise ValueError(
                f"Branch channels must sum to out_channels. Got {total_branch_channels} vs {out_channels}."
            )

        lifted_channels = in_channels + (2 if use_coordinate_grid else 0)
        self.input_projection = nn.Conv2d(lifted_channels, operator_width, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, operator_width),
            nn.GELU(),
            nn.Linear(operator_width, time_embedding_dim),
        )
        self.blocks = nn.ModuleList(
            [
                SharedOperatorBlock(
                    width=operator_width,
                    modes_height=spectral_modes_height,
                    modes_width=spectral_modes_width,
                    time_embedding_dim=time_embedding_dim,
                    dropout=dropout,
                )
                for _ in range(num_operator_layers)
            ]
        )
        self.trunk_norm = nn.GroupNorm(1, operator_width)
        self.trunk_proj = nn.Conv2d(operator_width, operator_width, kernel_size=1)

        self.branch_names = list(branch_channels.keys())
        self.branch_decoders = nn.ModuleDict(
            {
                name: BranchDecoder(
                    width=operator_width,
                    out_channels=int(branch_channels[name]),
                    hidden_channels=branch_hidden_channels,
                )
                for name in self.branch_names
            }
        )

        if predict_log_variance:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(operator_width, branch_hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(branch_hidden_channels, 1, kernel_size=1),
            )
        else:
            self.uncertainty_head = None

    def _coordinate_grid(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | int) -> ShellWeakRefineOperatorOutput:
        if isinstance(timestep, int):
            timestep = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0]).to(device=sample.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=sample.device, dtype=torch.long)

        x = sample
        if self.use_coordinate_grid:
            x = torch.cat(
                [x, self._coordinate_grid(sample.shape[0], sample.shape[-2], sample.shape[-1], sample.device)],
                dim=1,
            )

        x = self.input_projection(x)
        time_embedding = self.time_mlp(_timestep_embedding(timestep, self.time_embedding_dim))

        for block in self.blocks:
            x = block(x, time_embedding)

        x = self.trunk_norm(x)
        x = F.gelu(self.trunk_proj(x))

        sample_out = torch.cat([self.branch_decoders[name](x) for name in self.branch_names], dim=1)
        log_variance = None
        if self.uncertainty_head is not None:
            log_variance = torch.clamp(self.uncertainty_head(x), min=-6.0, max=3.0)

        return ShellWeakRefineOperatorOutput(sample=sample_out, log_variance=log_variance)
