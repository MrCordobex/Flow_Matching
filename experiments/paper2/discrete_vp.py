"""Faithful discrete VP sampler for the published Architect checkpoints.

The paper's Architect was trained with the Diffusers `DDPMScheduler` on integer
timesteps 0..999 and `squaredcos_cap_v2` betas. Its `alphas_cumprod` differs
from the analytic continuous cosine by up to 5e-4, and the network never saw a
non-integer timestep, so sampling is done here on the discrete grid rather than
through the continuous `CosineVPSchedule`. That removes both mismatches.

One sampler covers the whole stochasticity axis: eta=0 is DDIM, eta=1 is the
ancestral DDPM posterior, and anything in between interpolates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler

TRAIN_TIMESTEPS = 1000


@dataclass
class DiscreteVP:
    """Cosine VP schedule sampled on the integer grid the model was trained on."""

    alphas_cumprod: torch.Tensor  # (T,)

    @classmethod
    def from_scheduler_config(cls, config: dict) -> "DiscreteVP":
        scheduler = DDPMScheduler.from_config(config)
        return cls(alphas_cumprod=scheduler.alphas_cumprod.double())

    def to(self, device: torch.device) -> "DiscreteVP":
        return DiscreteVP(self.alphas_cumprod.to(device))

    def alpha_sigma(self, timestep: int) -> tuple[torch.Tensor, torch.Tensor]:
        """sqrt(alpha_bar) and sqrt(1 - alpha_bar) at an integer timestep."""
        bar = self.alphas_cumprod[timestep]
        return bar.sqrt(), (1.0 - bar).sqrt()

    def timesteps(self, steps: int) -> list[int]:
        """Descending integer grid, matching Diffusers' 'leading' spacing."""
        if not 1 <= steps <= TRAIN_TIMESTEPS:
            raise ValueError(f"steps must be in [1, {TRAIN_TIMESTEPS}]")
        ratio = TRAIN_TIMESTEPS // steps
        grid = (torch.arange(steps) * ratio).round().long().flip(0)
        return [int(value) for value in grid]


def split_velocity(
    state: torch.Tensor,
    velocity: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    clip_denoised: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean and noise estimates implied by a v prediction.

    After clipping x0 the noise is recomputed from x_t = alpha*x0 + sigma*eps so
    the pair stays consistent, which is what Diffusers does when clip_sample is
    on. The published sampler runs with clip_sample=True.
    """
    clean = alpha * state - sigma * velocity
    if clip_denoised:
        clean = clean.clamp(-1.0, 1.0)
        noise = (state - alpha * clean) / sigma.clamp_min(1e-12)
    else:
        noise = sigma * state + alpha * velocity
    return clean, noise


@torch.no_grad()
def sample(
    field: Callable[[torch.Tensor, int, int, int], torch.Tensor],
    initial: torch.Tensor,
    schedule: DiscreteVP,
    steps: int,
    eta: float = 0.0,
    clip_denoised: bool = True,
    generator: torch.Generator | None = None,
    step_callback: Callable[[int, int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Reverse the VP chain. `field` returns the (possibly guided) v prediction.

    `field` receives (state, timestep, step_index, total_steps) so a guidance
    schedule can depend on position along the trajectory.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    grid = schedule.timesteps(steps)
    state = initial
    for index, timestep in enumerate(grid):
        alpha_t, sigma_t = schedule.alpha_sigma(timestep)
        alpha_t = alpha_t.to(state.dtype)
        sigma_t = sigma_t.to(state.dtype)
        velocity = field(state, timestep, index, steps)
        clean, noise = split_velocity(state, velocity, alpha_t, sigma_t, clip_denoised)

        if index + 1 == steps:
            state = clean  # final step reports the denoised estimate
        else:
            previous = grid[index + 1]
            alpha_s, sigma_s = schedule.alpha_sigma(previous)
            alpha_s = alpha_s.to(state.dtype)
            sigma_s = sigma_s.to(state.dtype)
            # Standard DDIM/DDPM interpolation: eta=0 keeps the deterministic
            # direction, eta=1 recovers the ancestral posterior variance.
            stochastic = eta * (sigma_s / sigma_t) * (
                1.0 - (alpha_t / alpha_s).square()
            ).clamp_min(0.0).sqrt()
            direction = (sigma_s.square() - stochastic.square()).clamp_min(0.0).sqrt()
            state = alpha_s * clean + direction * noise
            if eta > 0:
                extra = torch.randn(state.shape, device=state.device,
                                    dtype=state.dtype, generator=generator)
                state = state + stochastic * extra
        if step_callback is not None:
            step_callback(index, timestep, state)
    return state
