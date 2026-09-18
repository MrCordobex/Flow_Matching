"""Continuous cosine VP diffusion with v prediction and DDIM sampling.

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


def vp_model_time(t: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(t, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    return value * TIME_SCALE


def sample_vp_path(clean: torch.Tensor, schedule: CosineVPSchedule) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x_t = alpha*x0 + sigma*noise and v = alpha*noise - sigma*x0."""
    batch_size = clean.shape[0]
    time = (torch.arange(batch_size, device=clean.device, dtype=clean.dtype)
            + torch.rand(batch_size, device=clean.device, dtype=clean.dtype)) / batch_size
    time = time[torch.randperm(batch_size, device=clean.device)]
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
) -> torch.Tensor:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    t_value = torch.as_tensor(t, device=state.device, dtype=state.dtype)
    next_value = torch.as_tensor(next_t, device=state.device, dtype=state.dtype)
    if bool(next_value >= t_value) or bool(next_value < 0):
        raise ValueError("DDIM requires 0 <= next_t < t")
    alpha_t, sigma_t = schedule.alpha_sigma(t_value)
    clean_hat = alpha_t * state - sigma_t * velocity
    if bool(next_value == 0):
        return clean_hat
    noise_hat = sigma_t * state + alpha_t * velocity
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
) -> torch.Tensor:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    grid = make_time_grid(steps, spacing, initial.device)
    state = initial
    for index in range(steps):
        current = float(grid[index])
        next_time = float(grid[index + 1])
        velocity = field(state, current)
        state = ddim_step(state, velocity, current, next_time, schedule, eta)
        if step_callback is not None:
            step_callback(index, next_time, state)
    return state
