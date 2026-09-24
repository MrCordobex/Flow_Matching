"""Load the published Architect / Engineer checkpoints for the paper-2 sweep.

These checkpoints carry `scheduler_config` (Diffusers DDPMScheduler) rather than
the `diffusion_config` the newer repository writes, so they are loaded here
instead of through `tfm_shells.sampling.guided.load_sampling_context`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfm_shells.models.factory import build_unet

from paper2.discrete_vp import DiscreteVP


@dataclass
class Surrogate:
    """A mechanical surrogate plus the statistics needed to read its output."""

    name: str
    model: torch.nn.Module
    stats: dict[str, Any]
    physics_mean: torch.Tensor
    physics_std: torch.Tensor


def _load(path: Path, device: torch.device) -> dict[str, Any]:
    # mmap keeps the 4 GB checkpoints (weights + optimizer) out of RAM until
    # the tensors that are actually used get touched.
    checkpoint = torch.load(path, map_location=device, weights_only=False, mmap=True)
    checkpoint.pop("optimizer_state_dict", None)
    return checkpoint


def load_architect(path: Path, device: torch.device) -> tuple[torch.nn.Module, DiscreteVP, dict]:
    checkpoint = _load(path, device)
    if checkpoint.get("role") != "architect":
        raise ValueError(f"{path} is not an architect checkpoint")
    config = checkpoint["model_config"]
    if config.get("prediction_type") != "v_prediction":
        raise ValueError("expected a v-prediction Architect")
    model = build_unet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    schedule = DiscreteVP.from_scheduler_config(checkpoint["scheduler_config"]).to(device)
    return model, schedule, checkpoint["normalization_stats"]


def load_surrogate(name: str, path: Path, device: torch.device) -> Surrogate:
    checkpoint = _load(path, device)
    role = checkpoint.get("role")
    if role != "engineer":
        # The clean-trained baseline may carry a different role string. The
        # state dict is what has to match, so this only warns.
        print(f"  note: {path.name} has role={role!r}, expected 'engineer'")
    model = build_unet(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    stats = checkpoint["normalization_stats"]
    mean = torch.tensor(np.asarray(stats["physics_mean"], dtype=np.float32),
                        device=device).unsqueeze(0)
    std = torch.tensor(np.asarray(stats["physics_std"], dtype=np.float32),
                       device=device).unsqueeze(0)
    return Surrogate(name=name, model=model, stats=stats, physics_mean=mean, physics_std=std)


def normalize_load(source: Path, stats: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Read the conditioning load field and map it to the Engineer's range."""
    with np.load(source) as data:
        field = np.asarray(data["fz"], dtype=np.float32)
    low, high = float(stats["fz_min"]), float(stats["fz_max"])
    scaled = 2.0 * (field - low) / (high - low + 1e-8) - 1.0
    return torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(device)


def architect_to_surrogate(
    state: torch.Tensor,
    architect_stats: dict[str, Any],
    surrogate_stats: dict[str, Any],
) -> torch.Tensor:
    """Affine, differentiable map between the two normalised height domains.

    Both published checkpoints share z in [0, 7.8509], so this is the identity
    for them; it is kept general so a differently normalised surrogate still works.
    """
    a_low, a_high = float(architect_stats["z_min"]), float(architect_stats["z_max"])
    s_low, s_high = float(surrogate_stats["z_min"]), float(surrogate_stats["z_max"])
    metres = 0.5 * (state + 1.0) * (a_high - a_low) + a_low
    return 2.0 * (metres - s_low) / (s_high - s_low + 1e-8) - 1.0


def denormalize_heights(state: torch.Tensor, architect_stats: dict[str, Any]) -> np.ndarray:
    low, high = float(architect_stats["z_min"]), float(architect_stats["z_max"])
    array = state.detach().to(torch.float32).cpu().numpy()
    return (0.5 * (array + 1.0) * (high - low) + low).astype(np.float32)
