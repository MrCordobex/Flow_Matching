from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from tfm_shells.constants import DISPLACEMENT_KEYS, FLEXION_KEYS, MEMBRANE_KEYS


def denormalize_physics(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
) -> torch.Tensor:
    return pred_norm * p_std + p_mean


def split_physics_channels(physics_real: torch.Tensor) -> dict[str, torch.Tensor]:
    idx = 0
    uz = physics_real[:, idx : idx + len(DISPLACEMENT_KEYS)]
    idx += len(DISPLACEMENT_KEYS)

    membrane = physics_real[:, idx : idx + len(MEMBRANE_KEYS)]
    idx += len(MEMBRANE_KEYS)

    flexion = physics_real[:, idx : idx + len(FLEXION_KEYS)]
    return {"uz": uz, "membrane": membrane, "flexion": flexion}


def branchwise_supervised_losses(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred = split_physics_channels(pred_norm)
    target = split_physics_channels(target_norm)
    return {
        "uz_mse": torch.mean((pred["uz"] - target["uz"]) ** 2),
        "membrane_mse": torch.mean((pred["membrane"] - target["membrane"]) ** 2),
        "flexion_mse": torch.mean((pred["flexion"] - target["flexion"]) ** 2),
    }


def compute_energy_terms_from_real_physics(
    physics_real: torch.Tensor,
    ds: torch.Tensor,
    dv: torch.Tensor,
    fz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = split_physics_channels(physics_real)
    uz = branches["uz"]
    membrane = branches["membrane"]
    flexion = branches["flexion"]

    se11, se22, se12 = membrane[:, 0:1], membrane[:, 1:2], membrane[:, 2:3]
    sf11, sf22, sf12 = membrane[:, 3:4], membrane[:, 4:5], membrane[:, 5:6]
    sk11, sk22, sk12 = flexion[:, 0:1], flexion[:, 1:2], flexion[:, 2:3]
    sm11, sm22, sm12 = flexion[:, 3:4], flexion[:, 4:5], flexion[:, 5:6]

    w_memb = (sf11 * se11 + sf22 * se22 + 2.0 * sf12 * se12) * ds
    w_flex = (sm11 * sk11 + sm22 * sk22 + 2.0 * sm12 * sk12) * ds
    w_ext = fz * uz * dv
    return w_memb, w_flex, w_ext


def compute_physical_residual(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
    ds: torch.Tensor,
    dv: torch.Tensor,
    fz: torch.Tensor,
) -> torch.Tensor:
    physics_real = denormalize_physics(pred_norm, p_mean, p_std)
    w_memb, w_flex, w_ext = compute_energy_terms_from_real_physics(physics_real, ds, dv, fz)
    delta_p = w_memb.sum(dim=(1, 2, 3)) + w_flex.sum(dim=(1, 2, 3)) - w_ext.sum(dim=(1, 2, 3))
    return delta_p.square()


def compute_energy_residual_map(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
    ds: torch.Tensor,
    dv: torch.Tensor,
    fz: torch.Tensor,
) -> torch.Tensor:
    physics_real = denormalize_physics(pred_norm, p_mean, p_std)
    w_memb, w_flex, w_ext = compute_energy_terms_from_real_physics(physics_real, ds, dv, fz)
    return w_memb + w_flex - w_ext


def compute_weak_form_residual(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
    ds: torch.Tensor,
    dv: torch.Tensor,
    fz: torch.Tensor,
    num_test_modes: int = 4,
) -> torch.Tensor:
    residual_map = compute_energy_residual_map(pred_norm, p_mean, p_std, ds, dv, fz)
    batch_size, _, height, width = residual_map.shape
    y = torch.linspace(0.0, 1.0, steps=height, device=residual_map.device)
    x = torch.linspace(0.0, 1.0, steps=width, device=residual_map.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    coefficients: list[torch.Tensor] = []
    area = ds.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    for mode_y in range(1, num_test_modes + 1):
        for mode_x in range(1, num_test_modes + 1):
            test_fn = (
                torch.sin(torch.pi * mode_x * grid_x) * torch.sin(torch.pi * mode_y * grid_y)
            ).unsqueeze(0).unsqueeze(0)
            coeff = (residual_map * test_fn * ds).sum(dim=(1, 2, 3)) / area
            coefficients.append(coeff)

    stacked = torch.stack(coefficients, dim=1)
    return stacked.square().mean(dim=1)


def _normalize_spatial_map(score: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    score = torch.abs(score)
    min_value = score.amin(dim=(2, 3), keepdim=True)
    max_value = score.amax(dim=(2, 3), keepdim=True)
    return (score - min_value) / (max_value - min_value + eps)


def build_active_refinement_mask(
    residual_map: torch.Tensor,
    uncertainty_map: torch.Tensor | None,
    topk_ratio: float,
    residual_weight: float,
    uncertainty_weight: float,
) -> torch.Tensor:
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError("topk_ratio must be in (0, 1].")

    residual_score = _normalize_spatial_map(residual_map.mean(dim=1, keepdim=True))
    if uncertainty_map is None:
        score = residual_weight * residual_score
    else:
        uncertainty_score = _normalize_spatial_map(uncertainty_map.mean(dim=1, keepdim=True))
        score = residual_weight * residual_score + uncertainty_weight * uncertainty_score

    batch_size, _, height, width = score.shape
    flat_score = score.view(batch_size, -1)
    num_selected = max(int(round(flat_score.shape[1] * topk_ratio)), 1)
    topk_indices = torch.topk(flat_score, k=num_selected, dim=1).indices

    mask = torch.zeros_like(flat_score)
    mask.scatter_(1, topk_indices, 1.0)
    return mask.view(batch_size, 1, height, width)


def compute_membrane_factor_map_from_real_physics(
    physics_real: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    branches = split_physics_channels(physics_real)
    membrane = branches["membrane"]
    flexion = branches["flexion"]

    se11, se22, se12 = membrane[:, 0:1], membrane[:, 1:2], membrane[:, 2:3]
    sf11, sf22, sf12 = membrane[:, 3:4], membrane[:, 4:5], membrane[:, 5:6]
    sk11, sk22, sk12 = flexion[:, 0:1], flexion[:, 1:2], flexion[:, 2:3]
    sm11, sm22, sm12 = flexion[:, 3:4], flexion[:, 4:5], flexion[:, 5:6]

    w_memb = sf11 * se11 + sf22 * se22 + 2.0 * sf12 * se12
    w_flex = sm11 * sk11 + sm22 * sk22 + 2.0 * sm12 * sk12
    mf = w_memb / (w_memb + w_flex + eps)
    return torch.clamp(mf, 0.0, 1.0)


def compute_membrane_factor_from_prediction(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
) -> torch.Tensor:
    physics_real = denormalize_physics(pred_norm, p_mean, p_std)
    return compute_membrane_factor_map_from_real_physics(physics_real)


def compute_mf_mean_numpy(mf_map: np.ndarray) -> float:
    return float(np.asarray(mf_map).mean())
