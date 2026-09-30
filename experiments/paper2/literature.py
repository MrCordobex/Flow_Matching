"""Physics-guidance methods from the literature, on the paper-2 sampler.

Every method samples the same frozen Architect on the same discrete VP grid
(eta = 1, clip_denoised = True), starts from the same shared noise pool, and
optimises the same objective J = (1 - mf)^2. Only the update rule differs, and
each rule follows its paper. Where a paper leaves a free constant this module
states the value used.

    method        paper                               where physics is evaluated
    ours          Martinez-Huertas et al. (paper 1)   R(x_t, t), bell schedule
    tc_detached   DPS as run in paper 1               R_clean(x0_hat), no Jacobian
    dps           Chung et al., ICLR 2023             R_clean(x0_hat), through the Architect
    mpgd          He et al., ICLR 2024                R_clean(x0_hat), step taken on x0_hat
    lgd           Song et al., ICML 2023 (LGD-MC)     E over x0_hat + r_t z, through the Architect
    dsg           Yang et al., ICML 2024              DPS gradient, step on the sampling sphere
    freedom       Yu et al., ICCV 2023                DPS gradient + time travel
    ugd           Bansal et al., ICLR 2024            forward + backward guidance + self-recurrence
    tfg           Ye et al., NeurIPS 2024             mean + variance guidance + recurrence
    bon           best-of-N                           final sample, scorer at t = 0
    fk            Singhal et al., ICML 2025           particle resampling on reward differences
    svdd          Li et al., 2024                     per-step selection by a value function
    dflow         Ben-Hamu et al., ICML 2024          optimise x_T through the whole chain
    terminal      post-hoc refinement (PhysGen/3DID)  gradient steps on the final geometry

Training-free methods use the clean surrogate: their premise is an off-the-shelf
predictor trained on clean data. The particle and value methods have a variant
whose value is the noise-aware Engineer on x_t (`scorer="na"`), because that
network is by construction a value function over noisy states.

Pairing: the initial noise is always the shared pool. Methods that draw a
different amount of randomness per step (particles, recurrence, candidates)
cannot share the ancestral stream with the base sampler; their per-chunk
generator is still seeded from the pool position, so reruns are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint

from paper2.discrete_vp import DiscreteVP, sample, split_velocity
from paper2.models import Surrogate, architect_to_surrogate
from paper2.providers import GuidanceCost, _membrane_factor, bell_weight, make_provider

ETA = 1.0
CLIP = True
BELL = dict(w_max=8.0, peak=0.5, width=0.22)
GRAD_CLIP = 5.0


@dataclass
class Context:
    architect: torch.nn.Module
    schedule: DiscreteVP
    stats: dict
    surrogates: dict[str, Surrogate]
    load: torch.Tensor  # (1, 1, H, W), already in the Engineer's normalised range


@dataclass
class Cost:
    """Network evaluations, split by whether a backward pass went through them."""

    architect: int = 0
    architect_grad: int = 0
    surrogate: int = 0
    surrogate_grad: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- primitives

def _t(x: torch.Tensor, timestep: int) -> torch.Tensor:
    return torch.full((x.shape[0],), float(timestep), device=x.device, dtype=torch.float32)


def velocity(ctx: Context, x: torch.Tensor, timestep: int, cost: Cost, grad: bool = False) -> torch.Tensor:
    if grad:
        cost.architect_grad += 1
        return ctx.architect(x, _t(x, timestep)).sample
    cost.architect += 1
    with torch.no_grad():
        return ctx.architect(x, _t(x, timestep)).sample


def coeffs(ctx: Context, timestep: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    alpha, sigma = ctx.schedule.alpha_sigma(timestep)
    return alpha.to(dtype), sigma.to(dtype)


def mf(ctx: Context, scorer: str, heights: torch.Tensor, timestep: int, cost: Cost,
       grad: bool = False) -> torch.Tensor:
    """Per-sample mean predicted MF of Architect-domain heights, seen at `timestep`."""
    surrogate = ctx.surrogates[scorer]
    mapped = architect_to_surrogate(heights, ctx.stats, surrogate.stats)
    field_ = ctx.load.expand(heights.shape[0], -1, -1, -1)
    if grad:
        cost.surrogate_grad += 1
        return _membrane_factor(surrogate, mapped, field_, _t(heights, timestep))
    cost.surrogate += 1
    with torch.no_grad():
        return _membrane_factor(surrogate, mapped, field_, _t(heights, timestep))


def objective(value: torch.Tensor) -> torch.Tensor:
    return (1.0 - value).square()


def transition(ctx: Context, x: torch.Tensor, clean: torch.Tensor, noise: torch.Tensor,
               index: int, grid: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of the step x_t -> x_s, exactly as discrete_vp.sample (eta = 1)."""
    if index + 1 == len(grid):
        return clean, torch.zeros((), device=x.device, dtype=x.dtype)
    alpha_t, sigma_t = coeffs(ctx, grid[index], x.dtype)
    alpha_s, sigma_s = coeffs(ctx, grid[index + 1], x.dtype)
    stochastic = ETA * (sigma_s / sigma_t) * (1.0 - (alpha_t / alpha_s).square()).clamp_min(0.0).sqrt()
    direction = (sigma_s.square() - stochastic.square()).clamp_min(0.0).sqrt()
    return alpha_s * clean + direction * noise, stochastic


def draw(mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if float(std) == 0.0:
        return mean
    return mean + std * torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)


def renoise(ctx: Context, x_s: torch.Tensor, index: int, grid: list[int],
            generator: torch.Generator) -> torch.Tensor:
    """Forward kernel q(x_t | x_s): the time-travel / self-recurrence step."""
    alpha_t, sigma_t = coeffs(ctx, grid[index], x_s.dtype)
    alpha_s, sigma_s = coeffs(ctx, grid[index + 1], x_s.dtype)
    ratio = alpha_t / alpha_s
    std = (sigma_t.square() - ratio.square() * sigma_s.square()).clamp_min(0.0).sqrt()
    return ratio * x_s + std * torch.randn(x_s.shape, device=x_s.device, dtype=x_s.dtype, generator=generator)


def per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (x.dim() - 1)))


def tweedie_gradient(ctx: Context, x: torch.Tensor, timestep: int, cost: Cost, scorer: str = "clean",
                     through: bool = True, mc: int = 0, smoothing: float = 0.0,
                     generator: torch.Generator | None = None, dps_norm: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """grad_x J(x0_hat(x)), plus the (clipped) clean and noise estimates of the step.

    through   backpropagate through the Architect (full DPS) or treat x0_hat as
              a detached function of x (the Jacobian is then alpha^-1 * I-free)
    mc        LGD-MC: average exp(-J) over x0_hat + r_t z, r_t = sigma_t
    smoothing TFG variance guidance: evaluate at x + smoothing * sigma_t * delta
    dps_norm  DPS step normalisation: divide by the residual |1 - mf|
    """
    alpha, sigma = coeffs(ctx, timestep, x.dtype)
    leaf = x.detach().requires_grad_(True)
    with torch.enable_grad():
        query = leaf
        if smoothing > 0.0:
            query = leaf + smoothing * sigma * torch.randn(leaf.shape, device=leaf.device,
                                                           dtype=leaf.dtype, generator=generator)
        v = velocity(ctx, query, timestep, cost, grad=through)
        if not through:
            v = v.detach()
        raw = alpha * query - sigma * v  # unclipped: clipping would zero the gradient
        if mc > 0:
            batch = raw.shape[0]
            repeated = raw.repeat_interleave(mc, dim=0)
            repeated = repeated + sigma * torch.randn(repeated.shape, device=raw.device,
                                                      dtype=raw.dtype, generator=generator)
            losses = objective(mf(ctx, scorer, repeated, 0, cost, grad=True)).view(batch, mc)
            loss = -(torch.logsumexp(-losses, dim=1) - math.log(mc))
            value = 1.0 - losses.mean(1).sqrt()
        else:
            value = mf(ctx, scorer, raw, 0, cost, grad=True)
            loss = objective(value)
        if dps_norm:
            loss = loss / loss.detach().sqrt().clamp_min(1e-6)
        gradient = torch.autograd.grad(loss.sum(), leaf)[0]
    with torch.no_grad():
        clean, noise = split_velocity(x, v.detach() if smoothing == 0.0 else velocity(ctx, x, timestep, cost),
                                      alpha, sigma, CLIP)
    return gradient.detach(), value.detach(), clean, noise


# ------------------------------------------------------------------- methods

def _bell_sample(ctx: Context, x: torch.Tensor, steps: int, generator: torch.Generator, cost: Cost,
                 provider_kind: str, engineer: str, strength: float) -> torch.Tensor:
    """The paper-1 sampler: v + sigma * clip(bell * gamma * grad). Identical to generate.py."""
    shared = GuidanceCost()
    provider = make_provider(provider_kind, ctx.architect, ctx.schedule, ctx.stats,
                             ctx.surrogates.get(engineer), ctx.surrogates.get("clean"),
                             ctx.load, CLIP, shared)

    def field_(state: torch.Tensor, timestep: int, index: int, total: int) -> torch.Tensor:
        v = velocity(ctx, state, timestep, cost)
        if strength <= 0.0:
            return v
        gradient, _ = provider(state, timestep)
        weight = bell_weight(index, total, BELL["w_max"], BELL["peak"], BELL["width"])
        correction = torch.clamp(weight * strength * gradient, -GRAD_CLIP, GRAD_CLIP)
        _, sigma = ctx.schedule.alpha_sigma(timestep)
        return v + sigma.to(v.dtype) * correction

    out = sample(field_, x, ctx.schedule, steps, eta=ETA, clip_denoised=CLIP, generator=generator)
    cost.architect += shared.architect
    cost.surrogate_grad += shared.surrogate
    return out


def ours(ctx, x, steps, generator, cost, strength, engineer="pbunet", **_):
    return _bell_sample(ctx, x, steps, generator, cost, "noise_aware", engineer, strength)


def tc_detached(ctx, x, steps, generator, cost, strength, **_):
    return _bell_sample(ctx, x, steps, generator, cost, "tweedie_clean", "pbunet", strength)


def unguided(ctx, x, steps, generator, cost, **_):
    return _bell_sample(ctx, x, steps, generator, cost, "noise_aware", "pbunet", 0.0)


def dps(ctx, x, steps, generator, cost, strength, **_):
    """x_{t-1} = x'_{t-1} - zeta * grad ||1 - mf(x0_hat)||, gradient through the denoiser."""
    grid = ctx.schedule.timesteps(steps)
    for index, timestep in enumerate(grid):
        gradient, _, clean, noise = tweedie_gradient(ctx, x, timestep, cost, dps_norm=True)
        mean, std = transition(ctx, x, clean, noise, index, grid)
        x = draw(mean, std, generator)
        if index + 1 < len(grid):
            x = x - strength * gradient
    return x


def mpgd(ctx, x, steps, generator, cost, strength, **_):
    """Gradient step on x0_hat itself (no Architect Jacobian), then the usual transition.

    MPGD without the autoencoder projection, the variant for pixel-space models;
    step c_t = strength * (1 - alpha_bar_t) as in the paper's time-dependent step.
    """
    grid = ctx.schedule.timesteps(steps)
    for index, timestep in enumerate(grid):
        alpha, sigma = coeffs(ctx, timestep, x.dtype)
        v = velocity(ctx, x, timestep, cost)
        clean, noise = split_velocity(x, v, alpha, sigma, CLIP)
        leaf = clean.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = objective(mf(ctx, "clean", leaf, 0, cost, grad=True))
            gradient = torch.autograd.grad(loss.sum(), leaf)[0]
        clean = (clean - strength * sigma.square() * gradient).clamp(-1.0, 1.0)
        mean, std = transition(ctx, x, clean, noise, index, grid)
        x = draw(mean, std, generator)
    return x


def lgd(ctx, x, steps, generator, cost, strength, particles=4, **_):
    """DPS update with the Monte Carlo loss -log mean_j exp(-J(x0_hat + sigma_t z_j))."""
    grid = ctx.schedule.timesteps(steps)
    for index, timestep in enumerate(grid):
        gradient, _, clean, noise = tweedie_gradient(ctx, x, timestep, cost, mc=particles,
                                                     generator=generator, dps_norm=True)
        mean, std = transition(ctx, x, clean, noise, index, grid)
        x = draw(mean, std, generator)
        if index + 1 < len(grid):
            x = x - strength * gradient
    return x


def dsg(ctx, x, steps, generator, cost, strength, **_):
    """Spherical Gaussian constraint: the guided step stays on the radius sqrt(n) * std shell.

    `strength` is the guidance rate g_r in [0, 1] mixing the sampled direction
    with the steepest-descent direction before projecting back onto the sphere.
    """
    grid = ctx.schedule.timesteps(steps)
    rate = min(max(strength, 0.0), 1.0)
    for index, timestep in enumerate(grid):
        gradient, _, clean, noise = tweedie_gradient(ctx, x, timestep, cost)
        mean, std = transition(ctx, x, clean, noise, index, grid)
        if float(std) == 0.0:
            x = mean
            continue
        dims = math.sqrt(x[0].numel())
        radius = dims * std
        sampled = std * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        steepest = -radius * gradient / per_sample_norm(gradient)
        mixed = sampled + rate * (steepest - sampled)
        x = mean + radius * mixed / per_sample_norm(mixed)
    return x


def freedom(ctx, x, steps, generator, cost, strength, repeats=3, window=(0.3, 0.7), **_):
    """DPS-type update with time travel: inside the window each step is redone `repeats` times."""
    grid = ctx.schedule.timesteps(steps)
    for index, timestep in enumerate(grid):
        progress = index / max(len(grid) - 1, 1)
        loops = repeats if (window[0] <= progress <= window[1] and index + 1 < len(grid)) else 1
        start = x
        for loop in range(loops):
            gradient, _, clean, noise = tweedie_gradient(ctx, start, timestep, cost, dps_norm=True)
            mean, std = transition(ctx, start, clean, noise, index, grid)
            x = draw(mean, std, generator)
            if index + 1 < len(grid):
                x = x - strength * gradient
            if loop + 1 < loops:
                start = renoise(ctx, x, index, grid, generator)
    return x


def ugd(ctx, x, steps, generator, cost, strength, recurrence=3, backward_steps=3, **_):
    """Universal guidance: forward (on eps), backward (on x0) and self-recurrence.

    Forward: eps <- eps + strength * sigma_t * grad_x J(x0_hat). Backward: m
    gradient steps on Delta for J(x0_hat + Delta) with step strength * sigma_t^2,
    folded back as eps <- eps - alpha/sigma * Delta. Every step is repeated
    `recurrence` times through q(x_t | x_{t-1}).
    """
    grid = ctx.schedule.timesteps(steps)
    for index, timestep in enumerate(grid):
        alpha, sigma = coeffs(ctx, timestep, x.dtype)
        loops = recurrence if index + 1 < len(grid) else 1
        start = x
        for loop in range(loops):
            gradient, _, clean, noise = tweedie_gradient(ctx, start, timestep, cost)
            noise = noise + strength * sigma * gradient
            clean = ((start - sigma * noise) / alpha).clamp(-1.0, 1.0)
            delta = torch.zeros_like(clean)
            for _ in range(backward_steps):
                leaf = (clean + delta).detach().requires_grad_(True)
                with torch.enable_grad():
                    loss = objective(mf(ctx, "clean", leaf, 0, cost, grad=True))
                    step = torch.autograd.grad(loss.sum(), leaf)[0]
                delta = delta - strength * sigma.square() * step
            noise = noise - (alpha / sigma.clamp_min(1e-6)) * delta
            clean = (clean + delta).clamp(-1.0, 1.0)
            mean, std = transition(ctx, start, clean, noise, index, grid)
            x = draw(mean, std, generator)
            if loop + 1 < loops:
                start = renoise(ctx, x, index, grid, generator)
    return x


def tfg(ctx, x, steps, generator, cost, strength, recurrence=2, iterations=2, smoothing=0.1,
        mean_ratio=1.0, **_):
    """Training-free guidance, one point of its hyperparameter space.

    Mean guidance: `iterations` steps on x0_hat with step mu_t. Variance guidance:
    the gradient of J(x0_hat(x_t + smoothing * sigma_t * delta)) through the
    Architect, step rho_t. Both scale with the noise level,
    rho_t = mu_t / mean_ratio = strength * (1 - alpha_bar_t), and every step is
    recurred `recurrence` times. The paper's beam search over this space is
    replaced by the common strength probe used for every method here.
    """
    grid = ctx.schedule.timesteps(steps)
    for index, timestep in enumerate(grid):
        alpha, sigma = coeffs(ctx, timestep, x.dtype)
        rho = strength * sigma.square()
        mu = mean_ratio * rho
        loops = recurrence if index + 1 < len(grid) else 1
        start = x
        for loop in range(loops):
            variance, _, clean, noise = tweedie_gradient(ctx, start, timestep, cost, smoothing=smoothing,
                                                         generator=generator)
            delta = torch.zeros_like(clean)
            for _ in range(iterations):
                leaf = (clean + delta).detach().requires_grad_(True)
                with torch.enable_grad():
                    loss = objective(mf(ctx, "clean", leaf, 0, cost, grad=True))
                    step = torch.autograd.grad(loss.sum(), leaf)[0]
                delta = delta - mu * step
            clean = (clean + delta).clamp(-1.0, 1.0)
            mean, std = transition(ctx, start, clean, noise, index, grid)
            x = draw(mean, std, generator)
            if index + 1 < len(grid):
                x = x - rho * variance
            if loop + 1 < loops:
                start = renoise(ctx, x, index, grid, generator)
    return x


def _score_final(ctx: Context, x0: torch.Tensor, scorer: str, cost: Cost) -> torch.Tensor:
    """Reward of finished samples: the scorer at t = 0 (the clean surrogate, or the Engineer at t = 0)."""
    name = "clean" if scorer == "clean" else scorer_engineer(scorer)
    return mf(ctx, name, x0, 0, cost)


def scorer_engineer(scorer: str) -> str:
    return "pbunet" if scorer == "na" else scorer.removeprefix("na:")


def _candidates(x: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """count initial states per slot; the first is the slot's own pool noise."""
    extra = torch.randn((x.shape[0] * (count - 1),) + tuple(x.shape[1:]), device=x.device,
                        dtype=x.dtype, generator=generator)
    stacked = torch.cat([x.unsqueeze(1), extra.view(x.shape[0], count - 1, *x.shape[1:])], dim=1)
    return stacked.flatten(0, 1)


def bon(ctx, x, steps, generator, cost, strength, scorer="clean", **_):
    """Best-of-N per slot: N unguided samples per pool position, keep the best by the scorer."""
    count = max(int(round(strength)), 1)
    starts = _candidates(x, count, generator)
    finals = _bell_sample(ctx, starts, steps, generator, cost, "noise_aware", "pbunet", 0.0)
    reward = _score_final(ctx, finals, scorer, cost).view(x.shape[0], count)
    best = reward.argmax(dim=1)
    return finals.view(x.shape[0], count, *x.shape[1:])[torch.arange(x.shape[0]), best]


def _intermediate_reward(ctx, x, timestep, cost, scorer, clean):
    if scorer == "clean":
        return mf(ctx, "clean", clean, 0, cost)  # reward on Tweedie's estimate
    return mf(ctx, scorer_engineer(scorer), x, timestep, cost)  # value of the noisy state itself


def fk(ctx, x, steps, generator, cost, strength, particles=4, scorer="clean", resample_every=None, **_):
    """Feynman-Kac steering, 'difference' potential G_t = exp(lambda * (r_t - r_{t+1})).

    `strength` is lambda. Particles resample within their pool slot, so the
    output keeps one sample per slot; the final pick is the highest reward.
    """
    grid = ctx.schedule.timesteps(steps)
    every = resample_every or max(len(grid) // 5, 1)
    batch = x.shape[0]
    state = _candidates(x, particles, generator)
    previous = torch.zeros(batch * particles, device=x.device)
    for index, timestep in enumerate(grid):
        alpha, sigma = coeffs(ctx, timestep, state.dtype)
        v = velocity(ctx, state, timestep, cost)
        clean, noise = split_velocity(state, v, alpha, sigma, CLIP)
        if index % every == 0 and index + 1 < len(grid):
            reward = _intermediate_reward(ctx, state, timestep, cost, scorer, clean)
            logits = (strength * (reward - previous)).view(batch, particles)
            pick = torch.multinomial(torch.softmax(logits, dim=1), particles, replacement=True,
                                     generator=generator)
            flat = (pick + torch.arange(batch, device=x.device).unsqueeze(1) * particles).flatten()
            state, clean, noise, previous = state[flat], clean[flat], noise[flat], reward[flat]
        mean, std = transition(ctx, state, clean, noise, index, grid)
        state = draw(mean, std, generator)
    final = _score_final(ctx, state, "clean" if scorer == "clean" else scorer, cost).view(batch, particles)
    best = final.argmax(dim=1)
    return state.view(batch, particles, *x.shape[1:])[torch.arange(batch), best]


def svdd(ctx, x, steps, generator, cost, strength, scorer="clean", **_):
    """Soft value-based decoding, alpha = 0: M candidate x_{t-1} per step, keep the best value.

    `strength` is M. scorer="clean" is SVDD-PM (reward on the candidate's
    Tweedie estimate, one extra Architect call per candidate); scorer="na" uses
    the noise-aware Engineer on the candidate x_{t-1} directly, as SVDD-MC's
    learned value would.
    """
    count = max(int(round(strength)), 1)
    grid = ctx.schedule.timesteps(steps)
    batch = x.shape[0]
    for index, timestep in enumerate(grid):
        alpha, sigma = coeffs(ctx, timestep, x.dtype)
        v = velocity(ctx, x, timestep, cost)
        clean, noise = split_velocity(x, v, alpha, sigma, CLIP)
        mean, std = transition(ctx, x, clean, noise, index, grid)
        if float(std) == 0.0 or count == 1:
            x = draw(mean, std, generator)
            continue
        options = mean.repeat_interleave(count, dim=0)
        options = options + std * torch.randn(options.shape, device=x.device, dtype=x.dtype, generator=generator)
        following = grid[index + 1]
        if scorer == "clean":
            a_s, s_s = coeffs(ctx, following, x.dtype)
            c_s, _ = split_velocity(options, velocity(ctx, options, following, cost), a_s, s_s, CLIP)
            value = mf(ctx, "clean", c_s, 0, cost)
        else:
            value = mf(ctx, scorer_engineer(scorer), options, following, cost)
        best = value.view(batch, count).argmax(dim=1)
        x = options.view(batch, count, *x.shape[1:])[torch.arange(batch), best]
    return x


def dflow(ctx, x, steps, generator, cost, strength, **_):
    """D-Flow: optimise the initial noise through the full chain with L-BFGS.

    `strength` is the number of L-BFGS iterations. The ancestral noise is fixed
    per iteration (same generator state), so the chain is a deterministic,
    differentiable map x_T -> x_0 for the stochastic eta = 1 sampler used by
    every other method. Each step is checkpointed to fit K = 100 in memory.
    """
    iterations = max(int(round(strength)), 0)
    grid = ctx.schedule.timesteps(steps)
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device))
    device = x.device

    def fixed_noise() -> list[torch.Tensor]:
        local = torch.Generator(device=device).manual_seed(seed)
        return [torch.randn(x.shape, device=device, dtype=x.dtype, generator=local) for _ in grid[:-1]]

    noises = fixed_noise()

    def one_step(state, index):
        timestep = grid[index]
        alpha, sigma = coeffs(ctx, timestep, state.dtype)
        v = ctx.architect(state, _t(state, timestep)).sample
        clean = (alpha * state - sigma * v).clamp(-1.0, 1.0)
        noise = (state - alpha * clean) / sigma.clamp_min(1e-12)
        mean, std = transition(ctx, state, clean, noise, index, grid)
        return mean + std * noises[index] if index + 1 < len(grid) else mean

    def chain(start):
        state = start
        for index in range(len(grid)):
            state = checkpoint(one_step, state, index, use_reentrant=False)
        return state

    if iterations == 0:
        with torch.no_grad():
            return chain(x)
    latent = x.detach().clone().requires_grad_(True)
    optimiser = torch.optim.LBFGS([latent], lr=1.0, max_iter=iterations, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        with torch.enable_grad():
            loss = objective(mf(ctx, "clean", chain(latent), 0, cost, grad=True)).sum()
            loss.backward()
        cost.architect_grad += len(grid)
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        cost.architect += len(grid)
        return chain(latent.detach())


def terminal(ctx, x, steps, generator, cost, strength, iterations=20, **_):
    """Unguided sampling, then gradient steps on the finished geometry (clean surrogate).

    The refinement stage of PhysGen / 3DID reduced to its common core: x0 <-
    clip(x0 - strength * grad J(x0)), `iterations` times.
    """
    final = _bell_sample(ctx, x, steps, generator, cost, "noise_aware", "pbunet", 0.0)
    for _ in range(iterations):
        leaf = final.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = objective(mf(ctx, "clean", leaf, 0, cost, grad=True))
            step = torch.autograd.grad(loss.sum(), leaf)[0]
        final = (final - strength * step).clamp(-1.0, 1.0)
    return final.detach()


METHODS: dict[str, Callable] = {
    "unguided": unguided,
    "ours": ours,
    "tc_detached": tc_detached,
    "dps": dps,
    "mpgd": mpgd,
    "lgd": lgd,
    "dsg": dsg,
    "freedom": freedom,
    "ugd": ugd,
    "tfg": tfg,
    "bon": bon,
    "fk": fk,
    "svdd": svdd,
    "dflow": dflow,
    "terminal": terminal,
}
