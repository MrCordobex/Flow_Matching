"""Linear conditional flow matching for normalized 64 x 64 shell height maps.

Time runs from Gaussian noise at 0 to the shell at 1. The UNet time embedding
uses 0..999 for compatibility with the existing Diffusers UNet2DModel.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

TIME_SCALE = 999.0


def model_time(t: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(t, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    return value * TIME_SCALE


def sample_flow_path(clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x_t, continuous t, and the target dx/dt = clean - noise."""
    noise = torch.randn_like(clean)
    t = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
    weight = t.view(-1, *([1] * (clean.ndim - 1)))
    state = (1.0 - weight) * noise + weight * clean
    return state, t, clean - noise


@torch.no_grad()
def integrate_flow(
    field: Callable[[torch.Tensor, float], torch.Tensor],
    initial: torch.Tensor,
    steps: int,
    solver: str = "euler",
    step_callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    state = initial
    dt = 1.0 / steps
    for index in range(steps):
        t = index * dt
        velocity = field(state, t)
        if solver == "heun":
            predictor = state + dt * velocity
            next_velocity = field(predictor, (index + 1) * dt)
            state = state + 0.5 * dt * (velocity + next_velocity)
        else:
            state = state + dt * velocity
        if step_callback is not None:
            step_callback(index, t + dt, state)
    return state
