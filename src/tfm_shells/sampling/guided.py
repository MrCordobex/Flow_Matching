"""Physics-guided DDIM sampling for the solid-shell cosine VP model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tfm_shells.utils.matplotlib_backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from tfm_shells.config import load_config, resolve_project_path, save_config
from tfm_shells.cosine_flow import (
    FLOW_METHOD,
    SOLVER_STAGES,
    CosineFlowPath,
    integrate_cosine_flow,
)
from tfm_shells.vp_diffusion import CosineVPSchedule, sample_ddim, sample_ddpm, vp_model_time
from tfm_shells.models.factory import build_unet
from tfm_shells.training.common import (
    bell_guidance_weight,
    make_run_name,
    polynomial_guidance_weight,
    prepare_run_directories,
    resolve_device,
    seed_everything,
)
from tfm_shells.utils.io import save_json
from tfm_shells.utils.physics import compute_membrane_factor_map_from_real_physics
from tfm_shells.utils.tracking import ExperimentTracker


@dataclass
class SamplingContext:
    architect: torch.nn.Module
    engineer: torch.nn.Module
    architect_stats: dict[str, Any]
    engineer_stats: dict[str, Any]
    fz_condition: torch.Tensor
    device: torch.device
    source_file: Path
    vp_schedule: CosineVPSchedule = field(default_factory=CosineVPSchedule)
    flow_path: CosineFlowPath = field(default_factory=CosineFlowPath)
    # 1.0 when the network already emits dx/dphi, 1/rate when it emits dx/dt.
    net_to_v: float = 1.0
    architect_method: str = "cosine_vp_v"


def _normalize(array: np.ndarray, low: float, high: float) -> np.ndarray:
    return (2.0 * (array - low) / (high - low + 1e-8) - 1.0).astype(np.float32)


def _architect_to_engineer(x: torch.Tensor, arch: dict, eng: dict) -> torch.Tensor:
    real = 0.5 * (x + 1.0) * (float(arch["z_max"]) - float(arch["z_min"])) + float(arch["z_min"])
    return 2.0 * (real - float(eng["z_min"])) / (float(eng["z_max"]) - float(eng["z_min"]) + 1e-8) - 1.0


def _check_vp_checkpoint(ckpt: dict, role: str, method: str | set[str]) -> dict[str, Any]:
    accepted = {method} if isinstance(method, str) else set(method)
    diffusion = ckpt.get("diffusion_config", {})
    if (diffusion.get("method") not in accepted or float(diffusion.get("time_scale", -1)) != 999.0
            or "cosine_s" not in diffusion):
        names = " or ".join(sorted(accepted))
        raise ValueError(f"{role} checkpoint must use the cosine VP schedule ({names}); retrain older flow checkpoints")
    return diffusion


def load_sampling_context(config: dict[str, Any], device: torch.device) -> SamplingContext:
    arch_path = resolve_project_path(config, config["architect"]["checkpoint"])
    eng_path = resolve_project_path(config, config["engineer"]["checkpoint"])
    arch_ckpt = torch.load(arch_path, map_location=device, weights_only=False)
    eng_ckpt = torch.load(eng_path, map_location=device, weights_only=False)
    arch_diffusion = _check_vp_checkpoint(arch_ckpt, "Architect", {"cosine_vp_v", FLOW_METHOD})
    eng_diffusion = _check_vp_checkpoint(eng_ckpt, "Engineer", "cosine_vp_conditioned")
    if float(arch_diffusion["cosine_s"]) != float(eng_diffusion["cosine_s"]):
        raise ValueError("Architect and Engineer checkpoints use different VP cosine schedules")
    if int(arch_ckpt["model_config"]["in_channels"]) != 1 or int(arch_ckpt["model_config"]["out_channels"]) != 1:
        raise ValueError("Architect checkpoint must use one input and one output channel")
    if (eng_ckpt["model_config"].get("kind") not in {"parallel_pb_unet", "hybrid_fourier_unet"}
            or int(eng_ckpt["model_config"]["in_channels"]) != 2
            or int(eng_ckpt["model_config"]["out_channels"]) != 13):
        raise ValueError("Engineer checkpoint must be a solid-only, two-input, three-branch surrogate")
    architect = build_unet(arch_ckpt["model_config"]).to(device)
    engineer = build_unet(eng_ckpt["model_config"]).to(device)
    architect.load_state_dict(arch_ckpt["model_state_dict"])
    engineer.load_state_dict(eng_ckpt["model_state_dict"])
    architect.eval()
    engineer.eval()
    source = resolve_project_path(config, config["conditioning"]["source_file"])
    with np.load(source) as data:
        fz = np.asarray(data["fz"], dtype=np.float32)
    if fz.shape != (1, int(arch_ckpt["model_config"]["sample_size"]), int(arch_ckpt["model_config"]["sample_size"])):
        raise ValueError(f"conditioning fz must have shape (1,H,W) at model resolution, got {fz.shape}")
    eng_stats = eng_ckpt["normalization_stats"]
    fz_normalized = _normalize(fz, float(eng_stats["fz_min"]), float(eng_stats["fz_max"]))
    fz_tensor = torch.from_numpy(fz_normalized).unsqueeze(0).to(device)
    cosine_s = float(arch_diffusion["cosine_s"])
    flow_path = CosineFlowPath(cosine_s)
    architect_method = str(arch_diffusion["method"])
    # Everything downstream works in dx/dphi units, so a flow network is divided
    # by the constant angular rate once, here.
    net_to_v = 1.0 / flow_path.rate if architect_method == FLOW_METHOD else 1.0
    return SamplingContext(architect, engineer, arch_ckpt["normalization_stats"], eng_stats,
                           fz_tensor, device, source, CosineVPSchedule(cosine_s),
                           flow_path, net_to_v, architect_method)


def _guide_weight(config: dict[str, Any], t: float) -> float:
    sampling = config["sampling"]
    if sampling.get("guidance_schedule", "bell") == "bell":
        return bell_guidance_weight(int(round(t * 1000)), 1001, float(sampling["guide_w_max"]), float(sampling["bell_peak"]), float(sampling["bell_width"]))
    if sampling["guidance_schedule"] == "poly":
        return polynomial_guidance_weight(int(round(t * 1000)), 1001, float(sampling["guide_w_min"]), float(sampling["guide_w_max"]), float(sampling["guide_power"]))
    raise ValueError("guidance_schedule must be 'bell' or 'poly'")


def _mf_from_engineer(context: SamplingContext, state: torch.Tensor, t: float) -> torch.Tensor:
    z_eng = _architect_to_engineer(state, context.architect_stats, context.engineer_stats)
    fz = context.fz_condition.expand(state.shape[0], -1, -1, -1)
    prediction = context.engineer(torch.cat([z_eng, fz], dim=1), vp_model_time(t, state.shape[0], context.device)).sample
    mean = torch.as_tensor(context.engineer_stats["physics_mean"], dtype=torch.float32, device=context.device).unsqueeze(0)
    std = torch.as_tensor(context.engineer_stats["physics_std"], dtype=torch.float32, device=context.device).unsqueeze(0)
    real = prediction * std + mean
    return compute_membrane_factor_map_from_real_physics(real).mean(dim=(1, 2, 3))


def evaluate_clean_mf(context: SamplingContext, states: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return _mf_from_engineer(context, states, 0.0).cpu().numpy()


def generate_samples(
    context: SamplingContext,
    config: dict[str, Any],
    initial: torch.Tensor,
    steps: int,
    solver: str = "ddim",
    show_progress: bool = False,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    if initial.ndim != 4 or initial.shape[1] != 1:
        raise ValueError("initial noise must have shape (B, 1, H, W)")
    if solver not in {"ddim", "ddpm"} | set(SOLVER_STAGES):
        raise ValueError(f"solver must be ddim, ddpm or one of {sorted(SOLVER_STAGES)}")
    scale = float(config["sampling"]["guidance_scale"])
    clip = float(config["sampling"]["grad_clip"])
    eta = float(config["sampling"].get("eta", 0.0))
    spacing = str(config["sampling"].get("time_spacing", "quadratic"))
    clip_denoised = bool(config["sampling"].get("clip_denoised", False))
    if clip <= 0 or scale < 0:
        raise ValueError("grad_clip must be positive and guidance_scale nonnegative")
    history: dict[str, list[float]] = {"t": [], "objective": [], "mf_mean": [], "grad_norm": [], "guide_weight": []}
    diagnostic: dict[str, float] = {}

    def velocity_field(state: torch.Tensor, t: float) -> torch.Tensor:
        """Guided dx/dphi. A flow checkpoint is rescaled to these units by net_to_v."""
        with torch.no_grad():
            raw = context.architect(state, vp_model_time(t, state.shape[0], context.device)).sample
            velocity = raw * context.net_to_v
        if scale == 0.0:
            diagnostic.update(objective=float("nan"), mf_mean=float("nan"), grad_norm=0.0, guide_weight=0.0)
            return velocity
        with torch.enable_grad():
            state_req = state.detach().requires_grad_(True)
            mf = _mf_from_engineer(context, state_req, t)
            per_sample = (1.0 - mf).square()
            grad = torch.autograd.grad(per_sample.sum(), state_req)[0]
        weight = _guide_weight(config, 1.0 - t)
        correction = torch.clamp(scale * weight * grad, -clip, clip)
        diagnostic.update(
            objective=float(per_sample.mean().detach()),
            mf_mean=float(mf.mean().detach()),
            grad_norm=float(correction.flatten(1).norm(dim=1).mean().detach()),
            guide_weight=weight,
        )
        _, sigma = context.vp_schedule.alpha_sigma(torch.as_tensor(t, device=state.device))
        return velocity + sigma * correction.detach()

    def flow_field(state: torch.Tensor, t: float) -> torch.Tensor:
        """dx/dt for the ODE solvers. Guidance rides along, already in dx/dphi."""
        return context.flow_path.velocity_from_v(velocity_field(state, t))

    progress = tqdm(total=steps, desc=f"cosine {solver} sampling", disable=not show_progress)

    def record(index: int, t: float, state: torch.Tensor) -> None:
        history["t"].append(t)
        for key in ("objective", "mf_mean", "grad_norm", "guide_weight"):
            history[key].append(diagnostic[key])
        progress.update(1)

    try:
        if solver == "ddpm":
            result = sample_ddpm(velocity_field, initial, steps, context.vp_schedule,
                                 spacing=spacing, step_callback=record,
                                 clip_denoised=clip_denoised)
        elif solver == "ddim":
            result = sample_ddim(velocity_field, initial, steps, context.vp_schedule,
                                 spacing=spacing, eta=eta, step_callback=record,
                                 clip_denoised=clip_denoised)
        else:
            result = integrate_cosine_flow(flow_field, initial, steps, context.flow_path,
                                           spacing=spacing, solver=solver,
                                           clip_denoised=clip_denoised, step_callback=record)
    finally:
        progress.close()
    return result, history


def denormalize_samples(context: SamplingContext, states: torch.Tensor) -> np.ndarray:
    low = float(context.architect_stats["z_min"])
    high = float(context.architect_stats["z_max"])
    return (0.5 * (states.detach().cpu().numpy() + 1.0) * (high - low) + low).astype(np.float32)


def run_guided_sampling(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    context = load_sampling_context(config, device)
    dirs = prepare_run_directories(config, role="sample")
    save_config(config, dirs["run_root"] / "config.yaml")
    batch_size = int(config["conditioning"]["batch_size"])
    height = int(context.architect.config.sample_size)
    initial = torch.randn((batch_size, 1, height, height), device=device)
    steps = int(config["sampling"]["num_inference_steps"])
    solver = str(config["sampling"].get("solver", "ddim"))
    states, history = generate_samples(context, config, initial, steps, solver, show_progress=True)
    samples = denormalize_samples(context, states)
    final_mf = evaluate_clean_mf(context, states)
    sample_file = dirs["run_root"] / "guided_samples.npz"
    np.savez_compressed(sample_file, z=samples)

    figure, axes = plt.subplots(2, int(np.ceil(batch_size / 2)), figsize=(16, 6))
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.axis("off")
        if index < batch_size:
            axis.imshow(samples[index, 0], cmap="viridis")
            axis.set_title(f"sample_{index}")
    figure.tight_layout()
    sample_png = dirs["run_root"] / "guided_samples.png"
    figure.savefig(sample_png, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, key in zip(axes, ("mf_mean", "objective", "grad_norm")):
        axis.plot(history["t"], history[key])
        axis.set_title(key)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    history_png = dirs["run_root"] / "guided_history.png"
    figure.savefig(history_png, dpi=180)
    plt.close(figure)

    stages = SOLVER_STAGES.get(solver, 1)
    summary = {
        "method": f"{context.architect_method}_{solver}",
        "architect_parameterization": "dx_dt" if context.net_to_v != 1.0 else "dx_dphi",
        "solver": solver,
        "eta": (1.0 if solver == "ddpm"
                else float(config["sampling"].get("eta", 0.0)) if solver == "ddim" else 0.0),
        "clip_denoised": bool(config["sampling"].get("clip_denoised", False)),
        "time_spacing": str(config["sampling"].get("time_spacing", "quadratic")),
        "num_inference_steps": steps,
        "architect_evaluations": steps * stages,
        "guidance_scale": float(config["sampling"]["guidance_scale"]),
        "samples_generated": batch_size,
        "sample_shape": list(samples.shape),
        "conditioning_source_file": str(context.source_file),
        "final_mf_mean": float(final_mf.mean()),
        "final_mf_std": float(final_mf.std()),
        "samples_file": str(sample_file),
    }
    save_json(summary, dirs["run_root"] / "summary.json")
    with ExperimentTracker(config, dirs["project_root"], make_run_name(config, role="sample")) as tracker:
        tracker.log_config(config)
        tracker.log_metrics({"samples_generated": float(batch_size), "final_mf_mean": summary["final_mf_mean"]})
        for path in (sample_file, sample_png, history_png, dirs["run_root"] / "summary.json"):
            tracker.log_artifact(path, artifact_path="samples")
    return summary
