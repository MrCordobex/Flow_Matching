"""Conditional flow matching on the cosine (trigonometric) interpolant.

The path is the same one the VP Architect already uses:

    phi(t) = (t + s) / (1 + s) * pi/2,  alpha = cos phi,  sigma = sin phi
    x_t = alpha * z + sigma * eps

Time 0 denotes a clean shell and time 1 denotes standard Gaussian noise, the
convention of the rest of this repository. Flow matching regresses the velocity
of that path with respect to *time*:

    u(x_t, t) = dx_t/dt = alpha' z + sigma' eps = k * (alpha eps - sigma z)

with the constant angular rate k = dphi/dt = pi / (2 (1 + s)). The bracket is
exactly the v target of `vp_diffusion`, so on this path the flow-matching
velocity and v prediction differ by the single constant k. That is why an
existing v checkpoint can be read as a flow field with `velocity_from_v`, and
why the Engineer stays compatible: the marginal states x_t and the 999t time
embedding are untouched.

Generation integrates dx/dt = u backwards from t=1 to t=0. `exponential` is the
closed-form step that treats (z_hat, eps_hat) as constant over the interval and
is therefore exact for this path; it coincides with deterministic DDIM. The
Runge-Kutta solvers are the generic alternatives and cost more evaluations per
step.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from tfm_shells.vp_diffusion import (
    TIME_SCALE,
    CosineVPSchedule,
    ddim_step,
    make_time_grid,
    sample_vp_path,
    vp_model_time,
)

# Evaluations of the velocity field per integration step.
SOLVER_STAGES = {"exponential": 1, "euler": 1, "midpoint": 2, "heun": 2, "rk4": 4}

FLOW_METHOD = "cosine_flow_u"


class CosineFlowPath:
    """Cosine interpolant expressed as a flow-matching probability path."""

    def __init__(self, s: float = 0.008) -> None:
        self.schedule = CosineVPSchedule(s)
        self.s = self.schedule.s
        self.rate = math.pi / (2.0 * (1.0 + self.s))

    def alpha_sigma(self, t: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        return self.schedule.alpha_sigma(t)

    def velocity_from_v(self, v: torch.Tensor) -> torch.Tensor:
        """dx/dt from dx/dphi."""
        return self.rate * v

    def v_from_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """dx/dphi from dx/dt, so VP samplers can consume a flow field."""
        return velocity / self.rate


def flow_model_time(t: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    """Share the 999t embedding with the VP Architect and the Engineer."""
    return vp_model_time(t, batch_size, device)


def sample_cosine_flow_path(
    clean: torch.Tensor,
    path: CosineFlowPath,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x_t, stratified t, and the flow-matching target dx_t/dt.

    The states and the time distribution are produced by `sample_vp_path`, so a
    flow Architect and a v Architect see byte-identical training batches for the
    same seed and differ only in the scale of the regression target.
    """
    state, time, v_target = sample_vp_path(clean, path.schedule)
    return state, time, path.velocity_from_v(v_target)


def denoised_from_velocity(
    state: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor | float,
    path: CosineFlowPath,
    clip_denoised: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean and noise estimates implied by a velocity, optionally clipped.

    After clipping z_hat the noise is recomputed from x_t = alpha z + sigma eps
    so the pair stays on the path, matching `vp_diffusion.ddim_step`.
    """
    alpha, sigma = path.alpha_sigma(torch.as_tensor(t, device=state.device, dtype=state.dtype))
    v = path.v_from_velocity(velocity)
    clean = alpha * state - sigma * v
    if clip_denoised:
        clean = clean.clamp(-1.0, 1.0)
        noise = (state - alpha * clean) / sigma.clamp_min(1e-6)
    else:
        noise = sigma * state + alpha * v
    return clean, noise


def project_velocity(
    state: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor | float,
    path: CosineFlowPath,
) -> torch.Tensor:
    """Rebuild the velocity after clamping the implied clean shell to [-1, 1].

    This is the identity when the estimate already lies inside the range, so it
    is the ODE-solver counterpart of `clip_denoised` in DDIM.
    """
    alpha, sigma = path.alpha_sigma(torch.as_tensor(t, device=state.device, dtype=state.dtype))
    clean, noise = denoised_from_velocity(state, velocity, t, path, clip_denoised=True)
    return path.velocity_from_v(alpha * noise - sigma * clean)


def exponential_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor | float,
    next_t: torch.Tensor | float,
    path: CosineFlowPath,
    clip_denoised: bool = False,
) -> torch.Tensor:
    """Exact update for constant (z_hat, eps_hat): x_next = alpha' z + sigma' eps."""
    return ddim_step(
        state,
        path.v_from_velocity(velocity),
        t,
        next_t,
        path.schedule,
        eta=0.0,
        clip_denoised=clip_denoised,
    )


@torch.no_grad()
def integrate_cosine_flow(
    field: Callable[[torch.Tensor, float], torch.Tensor],
    initial: torch.Tensor,
    steps: int,
    path: CosineFlowPath,
    spacing: str = "quadratic",
    solver: str = "heun",
    clip_denoised: bool = False,
    final_denoise: bool = True,
    step_callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Integrate dx/dt = field(x, t) from t=1 down to t=0 on the chosen grid.

    `final_denoise` returns the implied clean shell on the last interval instead
    of the integrated state. It matters: with a cosine offset s > 0 the path does
    not reach the data at t=0, it reaches alpha(0) z + sigma(0) eps. For s=0.008
    that leaves sigma(0) = 0.0125 of pure Gaussian noise in every pixel, which a
    faithful ODE solver reproduces and which lands directly on the curvature of
    the generated shell. DDIM hides this because its last step snaps to the clean
    estimate; keep this flag on so the ODE solvers agree with it.
    """
    if solver not in SOLVER_STAGES:
        raise ValueError(f"solver must be one of {sorted(SOLVER_STAGES)}")
    grid = make_time_grid(steps, spacing, initial.device)
    state = initial

    def evaluate(x: torch.Tensor, t: float) -> torch.Tensor:
        velocity = field(x, t)
        if clip_denoised and solver != "exponential":
            return project_velocity(x, velocity, t, path)
        return velocity

    for index in range(steps):
        current = float(grid[index])
        next_time = float(grid[index + 1])
        delta = next_time - current
        if next_time == 0.0 and final_denoise:
            # Same closing move as DDIM: report the denoised estimate, not the
            # state carrying sigma(0) worth of residual noise.
            state = denoised_from_velocity(
                state, evaluate(state, current), current, path, clip_denoised
            )[0]
        elif solver == "exponential":
            state = exponential_step(
                state, evaluate(state, current), current, next_time, path, clip_denoised
            )
        elif solver == "euler":
            state = state + delta * evaluate(state, current)
        elif solver == "midpoint":
            first = evaluate(state, current)
            middle = current + 0.5 * delta
            state = state + delta * evaluate(state + 0.5 * delta * first, middle)
        elif solver == "heun":
            first = evaluate(state, current)
            second = evaluate(state + delta * first, next_time)
            state = state + 0.5 * delta * (first + second)
        else:  # rk4
            middle = current + 0.5 * delta
            first = evaluate(state, current)
            second = evaluate(state + 0.5 * delta * first, middle)
            third = evaluate(state + 0.5 * delta * second, middle)
            fourth = evaluate(state + delta * third, next_time)
            state = state + (delta / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)
        if step_callback is not None:
            step_callback(index, next_time, state)
    return state


__all__ = [
    "FLOW_METHOD",
    "SOLVER_STAGES",
    "TIME_SCALE",
    "CosineFlowPath",
    "denoised_from_velocity",
    "exponential_step",
    "flow_model_time",
    "integrate_cosine_flow",
    "project_velocity",
    "sample_cosine_flow_path",
]
