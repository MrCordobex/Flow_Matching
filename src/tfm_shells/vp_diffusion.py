"""Continuous cosine VP diffusion with v prediction and DDIM/DDPM sampling.

Time 0 denotes a clean shell; time 1 denotes standard Gaussian noise.
The v target equals the derivative of the VP path with respect to its angle.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


TIME_SCALE = 999.0


class CosineVPSchedule:
    def __init__(self, s: float = 0.008) -> None:
        if s < 0:
            raise ValueError("cosine offset s must be nonnegative")
        self.s = float(s)

    def alpha_sigma(self, t: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(t)
        angle = (value + self.s) / (1.0 + self.s) * (math.pi / 2.0)
        return torch.cos(angle).clamp_min(0.0), torch.sin(angle).clamp_min(0.0)


class DiscreteCosineSchedule:
    """The integer-grid cosine schedule of the Diffusers DDPMScheduler.

    The published Architect was trained on integer timesteps 0..999 with
    `squaredcos_cap_v2` betas, whose alpha_bar differs from the analytic cosine
    by up to 5e-4. A surrogate that guides that Architect should see the same
    states, so time t in [0, 1] is snapped to the integer grid t*999.
    """

    def __init__(self, s: float = 0.008, steps: int = 1000) -> None:
        from diffusers import DDPMScheduler

        self.s = float(s)
        self.steps = int(steps)
        scheduler = DDPMScheduler(num_train_timesteps=self.steps, beta_schedule="squaredcos_cap_v2")
        self.alphas_cumprod = scheduler.alphas_cumprod.double()

    def snap(self, t: torch.Tensor) -> torch.Tensor:
        """Round t in [0, 1] to the nearest integer timestep, still expressed in [0, 1]."""
        return torch.round(t * TIME_SCALE).clamp(0, self.steps - 1) / TIME_SCALE

    def alpha_sigma(self, t: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(t)
        index = torch.round(value * TIME_SCALE).long().clamp(0, self.steps - 1)
        bar = self.alphas_cumprod.to(index.device)[index].to(value.dtype if value.is_floating_point()
                                                             else torch.float32)
        return bar.sqrt(), (1.0 - bar).clamp_min(0.0).sqrt()


def vp_model_time(t: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(t, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    return value * TIME_SCALE


def sample_vp_path(
    clean: torch.Tensor,
    schedule: CosineVPSchedule | DiscreteCosineSchedule,
    time_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x_t = alpha*x0 + sigma*noise and v = alpha*noise - sigma*x0.

    Times are stratified over the batch. `time_power` > 1 maps u -> u**power,
    which concentrates training on nearly clean states (power 2 puts 45% of the
    samples at t <= 0.2 instead of 20%) while still covering the noisy end.
    """
    batch_size = clean.shape[0]
    time = (torch.arange(batch_size, device=clean.device, dtype=clean.dtype)
            + torch.rand(batch_size, device=clean.device, dtype=clean.dtype)) / batch_size
    time = time[torch.randperm(batch_size, device=clean.device)]
    if time_power != 1.0:
        time = time.pow(time_power)
    if isinstance(schedule, DiscreteCosineSchedule):
        time = schedule.snap(time)
    alpha, sigma = schedule.alpha_sigma(time)
    shape = (-1, *([1] * (clean.ndim - 1)))
    alpha, sigma = alpha.reshape(shape), sigma.reshape(shape)
    noise = torch.randn_like(clean)
    return alpha * clean + sigma * noise, time, alpha * noise - sigma * clean


def make_time_grid(steps: int, spacing: str, device: torch.device) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if spacing == "uniform":
        return 1.0 - u
    if spacing == "quadratic":
        return (1.0 - u).square()
    raise ValueError("spacing must be 'uniform' or 'quadratic'")


def ddim_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor | float,
    next_t: torch.Tensor | float,
    schedule: CosineVPSchedule,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
    clip_denoised: bool = False,
) -> torch.Tensor:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    t_value = torch.as_tensor(t, device=state.device, dtype=state.dtype)
    next_value = torch.as_tensor(next_t, device=state.device, dtype=state.dtype)
    if bool(next_value >= t_value) or bool(next_value < 0):
        raise ValueError("DDIM requires 0 <= next_t < t")
    alpha_t, sigma_t = schedule.alpha_sigma(t_value)
    clean_hat = alpha_t * state - sigma_t * velocity
    if clip_denoised:
        clean_hat = clean_hat.clamp(-1.0, 1.0)
    if bool(next_value == 0):
        return clean_hat
    # After clipping x0, recompute epsilon so x_t = alpha_t*x0 + sigma_t*epsilon
    # remains true. With no clipping this equals sigma_t*state + alpha_t*velocity.
    noise_hat = ((state - alpha_t * clean_hat) / sigma_t
                 if clip_denoised else sigma_t * state + alpha_t * velocity)
    alpha_next, sigma_next = schedule.alpha_sigma(next_value)
    stochastic = eta * (sigma_next / sigma_t) * torch.sqrt(
        (1.0 - (alpha_t / alpha_next).square()).clamp_min(0.0)
    )
    direction = torch.sqrt((sigma_next.square() - stochastic.square()).clamp_min(0.0))
    result = alpha_next * clean_hat + direction * noise_hat
    if eta > 0:
        result = result + stochastic * (torch.randn_like(state) if noise is None else noise)
    return result


@torch.no_grad()
def sample_ddim(
    field: Callable[[torch.Tensor, float], torch.Tensor],
    initial: torch.Tensor,
    steps: int,
    schedule: CosineVPSchedule,
    spacing: str = "quadratic",
    eta: float = 0.0,
    step_callback: Callable[[int, float, torch.Tensor], None] | None = None,
    clip_denoised: bool = False,
) -> torch.Tensor:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    grid = make_time_grid(steps, spacing, initial.device)
    state = initial
    for index in range(steps):
        current = float(grid[index])
        next_time = float(grid[index + 1])
        velocity = field(state, current)
        state = ddim_step(state, velocity, current, next_time, schedule, eta,
                          clip_denoised=clip_denoised)
        if step_callback is not None:
            step_callback(index, next_time, state)
    return state


@torch.no_grad()
def sample_ddpm(
    field: Callable[[torch.Tensor, float], torch.Tensor],
    initial: torch.Tensor,
    steps: int,
    schedule: CosineVPSchedule,
    spacing: str = "quadratic",
    step_callback: Callable[[int, float, torch.Tensor], None] | None = None,
    clip_denoised: bool = False,
) -> torch.Tensor:
    """Ancestral VP posterior on the selected grid (DDIM eta=1).

    This shares the exact cosine path and v network with deterministic DDIM.
    It is not the discrete DDPMScheduler used by the earlier TFM checkpoint.
    """
    return sample_ddim(field, initial, steps, schedule, spacing=spacing, eta=1.0,
                       step_callback=step_callback, clip_denoised=clip_denoised)
