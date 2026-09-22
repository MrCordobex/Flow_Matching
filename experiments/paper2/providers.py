"""Gradient providers for Membrane-Factor guidance.

Each provider answers the same question — "which way should x_t move to raise
the Membrane Factor?" — but differs in *where* the mechanical operator is
evaluated. That is the axis the published paper argues about, and the one this
sweep pushes into the few-step regime.

    noise_aware    R(x_t, t)            the paper's time-conditioned Engineer
    tweedie_clean  R_clean(x0_hat, 0)   the DPS baseline: a separately trained
                                        clean surrogate on Tweedie's estimate
    tweedie_self   R(x0_hat, 0)         the same noise-aware weights queried on
                                        the denoised estimate, which isolates
                                        *where* you evaluate from *what* you
                                        trained on
    naive_clean    R_clean(x_t, 0)      a clean surrogate fed the noisy state

Following the published implementation, the Architect output inside the Tweedie
providers is detached, so no gradient flows back through the denoiser. That is
the standard practical form of DPS and keeps every provider at one Architect
evaluation per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from tfm_shells.utils.physics import compute_membrane_factor_map_from_real_physics

from paper2.discrete_vp import DiscreteVP, split_velocity
from paper2.models import Surrogate, architect_to_surrogate

PROVIDERS = ("noise_aware", "tweedie_clean", "tweedie_self", "naive_clean")


@dataclass
class GuidanceCost:
    """Model evaluations consumed, so cost can be reported per provider."""

    architect: int = 0
    surrogate: int = 0


def bell_weight(step_index: int, total_steps: int, w_max: float,
                peak: float, width: float) -> float:
    """Gaussian schedule over reverse progress, Eq. (62) of the paper."""
    if total_steps <= 1:
        return float(w_max)
    progress = step_index / float(total_steps - 1)
    return float(w_max * math.exp(-((progress - peak) ** 2) / (2.0 * width ** 2)))


def _membrane_factor(surrogate: Surrogate, heights: torch.Tensor,
                     load: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    prediction = surrogate.model(torch.cat([heights, load], dim=1), timestep).sample
    real = prediction * surrogate.physics_std + surrogate.physics_mean
    return compute_membrane_factor_map_from_real_physics(real).mean(dim=(1, 2, 3))


def make_provider(
    kind: str,
    architect: torch.nn.Module,
    schedule: DiscreteVP,
    architect_stats: dict,
    engineer: Surrogate | None,
    clean: Surrogate | None,
    load: torch.Tensor,
    clip_denoised: bool,
    cost: GuidanceCost,
) -> Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]:
    """Return f(state, timestep) -> (gradient wrt state, mean predicted MF)."""
    if kind not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}")
    if kind in ("noise_aware", "tweedie_self") and engineer is None:
        raise ValueError(f"provider {kind} needs the time-conditioned Engineer")
    if kind in ("tweedie_clean", "naive_clean") and clean is None:
        raise ValueError(f"provider {kind} needs the clean surrogate checkpoint")

    def provider(state: torch.Tensor, timestep: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        device = state.device
        requires = state.detach().clone().requires_grad_(True)
        zeros = torch.zeros(batch, device=device, dtype=torch.float32)
        current = torch.full((batch,), float(timestep), device=device, dtype=torch.float32)
        field = load.expand(batch, -1, -1, -1)

        with torch.enable_grad():
            if kind == "noise_aware":
                heights = architect_to_surrogate(requires, architect_stats, engineer.stats)
                mf = _membrane_factor(engineer, heights, field, current)
                cost.surrogate += 1
            elif kind == "naive_clean":
                heights = architect_to_surrogate(requires, architect_stats, clean.stats)
                mf = _membrane_factor(clean, heights, field, zeros)
                cost.surrogate += 1
            else:
                # Tweedie: denoise first, then evaluate the operator at t = 0.
                with torch.no_grad():
                    velocity = architect(requires, current).sample
                cost.architect += 1
                alpha, sigma = schedule.alpha_sigma(timestep)
                estimate, _ = split_velocity(requires, velocity, alpha.to(state.dtype),
                                             sigma.to(state.dtype), clip_denoised)
                target = engineer if kind == "tweedie_self" else clean
                heights = architect_to_surrogate(estimate, architect_stats, target.stats)
                mf = _membrane_factor(target, heights, field, zeros)
                cost.surrogate += 1

            objective = (1.0 - mf).square()
            gradient = torch.autograd.grad(objective.sum(), requires)[0]
        return gradient.detach(), mf.detach()

    return provider
