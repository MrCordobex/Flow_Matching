from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EquiNOOutput:
    sample: torch.Tensor


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


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_height: int,
        modes_width: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width

        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight_top = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_height, modes_width, 2)
        )
        self.weight_bottom = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_height, modes_width, 2)
        )

    @staticmethod
    def _complex_mul2d(x_ft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x_ft, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes_height = min(self.modes_height, height)
        modes_width = min(self.modes_width, width // 2 + 1)

        weight_top = torch.view_as_complex(self.weight_top[:, :, :modes_height, :modes_width].contiguous())
        weight_bottom = torch.view_as_complex(
            self.weight_bottom[:, :, :modes_height, :modes_width].contiguous()
        )

        out_ft[:, :, :modes_height, :modes_width] = self._complex_mul2d(
            x_ft[:, :, :modes_height, :modes_width],
            weight_top,
        )
        out_ft[:, :, -modes_height:, :modes_width] = self._complex_mul2d(
            x_ft[:, :, -modes_height:, :modes_width],
            weight_bottom,
        )

        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class EquiNOBlock(nn.Module):
    def __init__(
        self,
        width: int,
        modes_height: int,
        modes_width: int,
        time_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.GroupNorm(1, width)
        self.post_norm = nn.GroupNorm(1, width)
        self.spectral = SpectralConv2d(width, width, modes_height, modes_width)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.time_proj = nn.Linear(time_embedding_dim, width)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_bias = self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)

        residual = x
        x = self.pre_norm(x)
        x = self.spectral(x) + self.pointwise(x) + time_bias
        x = F.gelu(x)
        x = x + residual

        residual = x
        x = self.post_norm(x)
        x = self.channel_mlp(x)
        x = F.gelu(x)
        return x + residual


class ModalBranchHead(nn.Module):
    def __init__(
        self,
        width: int,
        out_channels: int,
        sample_size: int,
        hidden_channels: int,
        modal_rank: int,
        modal_residual_weight: float,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.modal_rank = modal_rank
        self.modal_residual_weight = modal_residual_weight

        self.local_head = nn.Sequential(
            nn.Conv2d(width, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        if modal_rank > 0:
            self.coeff_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(width, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels * modal_rank),
            )
            self.modal_basis = nn.Parameter(
                0.02 * torch.randn(out_channels, modal_rank, sample_size, sample_size)
            )
        else:
            self.coeff_head = None
            self.modal_basis = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.local_head(x)
        if self.modal_rank <= 0 or self.coeff_head is None or self.modal_basis is None:
            return out

        coefficients = self.coeff_head(x).view(x.shape[0], self.out_channels, self.modal_rank)
        modal = torch.einsum("bcr,crhw->bchw", coefficients, self.modal_basis)
        return out + self.modal_residual_weight * modal


class EquiNOModel(nn.Module):
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
        head_hidden_channels: int = 128,
        modal_rank: int = 12,
        modal_residual_weight: float = 0.25,
        branch_channels: dict[str, int] | None = None,
        use_coordinate_grid: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_coordinate_grid = use_coordinate_grid
        self.time_embedding_dim = time_embedding_dim

        lifted_channels = in_channels + (2 if use_coordinate_grid else 0)
        self.input_projection = nn.Conv2d(lifted_channels, operator_width, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, operator_width),
            nn.GELU(),
            nn.Linear(operator_width, time_embedding_dim),
        )
        self.blocks = nn.ModuleList(
            [
                EquiNOBlock(
                    width=operator_width,
                    modes_height=spectral_modes_height,
                    modes_width=spectral_modes_width,
                    time_embedding_dim=time_embedding_dim,
                    dropout=dropout,
                )
                for _ in range(num_operator_layers)
            ]
        )
        self.output_norm = nn.GroupNorm(1, operator_width)
        self.output_projection = nn.Conv2d(operator_width, operator_width, kernel_size=1)

        branch_channels = branch_channels or {}
        if branch_channels:
            total_branch_channels = sum(int(value) for value in branch_channels.values())
            if total_branch_channels != out_channels:
                raise ValueError(
                    f"Branch channels must sum to out_channels. Got {total_branch_channels} vs {out_channels}."
                )
            self.branch_names = list(branch_channels.keys())
            self.branch_heads = nn.ModuleDict(
                {
                    name: ModalBranchHead(
                        width=operator_width,
                        out_channels=int(branch_channels[name]),
                        sample_size=sample_size,
                        hidden_channels=head_hidden_channels,
                        modal_rank=modal_rank,
                        modal_residual_weight=modal_residual_weight,
                    )
                    for name in self.branch_names
                }
            )
            self.single_head = None
        else:
            self.branch_names = []
            self.branch_heads = nn.ModuleDict()
            self.single_head = ModalBranchHead(
                width=operator_width,
                out_channels=out_channels,
                sample_size=sample_size,
                hidden_channels=head_hidden_channels,
                modal_rank=modal_rank,
                modal_residual_weight=modal_residual_weight,
            )

    def _coordinate_grid(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | int) -> EquiNOOutput:
        if isinstance(timestep, int):
            timestep = torch.full(
                (sample.shape[0],),
                timestep,
                device=sample.device,
                dtype=torch.long,
            )
        elif timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0]).to(device=sample.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=sample.device, dtype=torch.long)

        x = sample
        if self.use_coordinate_grid:
            grid = self._coordinate_grid(sample.shape[0], sample.shape[-2], sample.shape[-1], sample.device)
            x = torch.cat([x, grid], dim=1)

        x = self.input_projection(x)
        time_embedding = _timestep_embedding(timestep, self.time_embedding_dim)
        time_embedding = self.time_mlp(time_embedding)

        for block in self.blocks:
            x = block(x, time_embedding)

        x = self.output_norm(x)
        x = F.gelu(self.output_projection(x))

        if self.single_head is not None:
            output = self.single_head(x)
        else:
            output = torch.cat([self.branch_heads[name](x) for name in self.branch_names], dim=1)

        return EquiNOOutput(sample=output)
