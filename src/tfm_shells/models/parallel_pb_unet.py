from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers import UNet2DModel


@dataclass
class ParallelUNetOutput:
    sample: torch.Tensor


class ParallelPBUNet(nn.Module):
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        layers_per_block: int,
        block_out_channels: tuple[int, ...],
        down_block_types: tuple[str, ...],
        up_block_types: tuple[str, ...],
        branch_channels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        branch_channels = branch_channels or {
            "u": 1,
            "m": 6,
            "f": 6,
        }

        common_kwargs: dict[str, Any] = {
            "sample_size": sample_size,
            "in_channels": in_channels,
            "layers_per_block": layers_per_block,
            "block_out_channels": block_out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
        }

        self.unet_u = UNet2DModel(out_channels=int(branch_channels["u"]), **common_kwargs)
        self.unet_m = UNet2DModel(out_channels=int(branch_channels["m"]), **common_kwargs)
        self.unet_f = UNet2DModel(out_channels=int(branch_channels["f"]), **common_kwargs)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | int) -> ParallelUNetOutput:
        pred_u = self.unet_u(sample, timestep).sample
        pred_m = self.unet_m(sample, timestep).sample
        pred_f = self.unet_f(sample, timestep).sample
        return ParallelUNetOutput(sample=torch.cat([pred_u, pred_m, pred_f], dim=1))
