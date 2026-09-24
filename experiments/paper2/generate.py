"""GPU stage: generate every geometry the sampling-budget study needs.

No mechanics are judged here. This writes elevation maps in metres plus the
metadata each run needs to be reproduced and costed; `evaluate_all.py` then runs
Kratos over them on CPU, and `analyze.py` joins the two.

    python -m paper2.generate --models-dir <dir> --output <dir> [--blocks b1 b2]

Runs are skipped when their output already exists, so the sweep can be resumed
after a disconnect.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paper2 import matrix
from paper2.discrete_vp import DiscreteVP, sample
from paper2.matrix import Run
from paper2.models import (
    denormalize_heights,
    load_architect,
    load_surrogate,
    normalize_load,
)
from paper2.providers import GuidanceCost, bell_weight, make_provider

SEED = matrix.DEFAULT_SEED
SNAPSHOTS = 20  # full correction fields kept per sample, spread over the guided steps


class GuidanceRecorder:
    """Per-sample, per-step record of the guidance one batch receives.

    `d_t = -sigma_t^2 * g_t` is the shift a step's clipped correction g_t induces
    in the clean estimate: g is the gradient of the loss (1 - mf)^2, and adding
    sigma * g to v moves x0_hat = alpha * x - sigma * v by -sigma^2 * g. Its running vector
    sum and path length give the straightness of the guidance, see PREREGISTRO.md.
    """

    def __init__(self, snapshot_steps: set[int], grad_clip: float) -> None:
        self.snapshot_steps, self.grad_clip = snapshot_steps, grad_clip
        self.previous = self.cumulative = self.path = None
        self.columns: dict[str, list[np.ndarray]] = {
            k: [] for k in ("norm_g", "norm_d", "cos_prev", "clip_frac", "mf_pred")}
        self.timesteps: list[int] = []
        self.weights: list[float] = []
        self.snapshots: list[np.ndarray] = []

    def add(self, index: int, timestep: int, weight: float, correction: torch.Tensor,
            mf: torch.Tensor, sigma: float) -> None:
        g = correction.detach().float().flatten(1)
        d = -g * sigma ** 2
        if self.cumulative is None:
            self.cumulative = torch.zeros_like(d)
            self.path = torch.zeros(d.shape[0], device=d.device)
        norm_d = d.norm(dim=1)
        self.cumulative += d
        self.path += norm_d
        cos = (F.cosine_similarity(g, self.previous, dim=1) if self.previous is not None
               else torch.full_like(norm_d, float("nan")))
        self.previous = g
        for key, value in (("norm_g", g.norm(dim=1)), ("norm_d", norm_d), ("cos_prev", cos),
                           ("clip_frac", (g.abs() >= self.grad_clip * (1 - 1e-6)).float().mean(1)),
                           ("mf_pred", mf.float())):
            self.columns[key].append(value.cpu().numpy())
        self.timesteps.append(int(timestep))
        self.weights.append(float(weight))
        if index in self.snapshot_steps:
            self.snapshots.append(d.half().cpu().numpy())

    def result(self, size: int) -> dict[str, np.ndarray]:
        data = {key: np.stack(values, axis=1) for key, values in self.columns.items()}
        data["cumulative"] = self.cumulative.cpu().numpy().reshape(-1, size, size)
        data["path"] = self.path.cpu().numpy()
        data["snapshots"] = np.stack(self.snapshots, axis=1).reshape(len(data["path"]), -1, size, size)
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="holds architect_solid.pt, engineer_solid.pt and optionally clean_solid.pt")
    parser.add_argument("--conditioning", type=Path, required=True,
                        help="npz with the fz field, e.g. shell_2600.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", nargs="*", default=None, help=f"subset of {sorted(matrix.BLOCKS)}")
    parser.add_argument("--only", nargs="*", default=None, help="keep runs whose run_id contains any of these")
    parser.add_argument("--samples", type=int, default=100)
    # 25 keeps the ancestral noise paired with b1-b5: each chunk of the pool is seeded
    # from its start position, so a different batch size draws different noise.
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initial_noise(samples: int, size: int, device: torch.device, seed: int = SEED) -> torch.Tensor:
    """One shared pool of initial states, so every run is paired."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(samples, 1, size, size, generator=generator).to(device)


def run_one(
    run: Run,
    noise: torch.Tensor,
    architect: torch.nn.Module,
    schedule: DiscreteVP,
    architect_stats: dict,
    surrogates: dict,
    load: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    engineer = surrogates.get("pbunet" if run.engineer == "pbunet" else "hybrid")
    clean = surrogates.get("clean")
    cost = GuidanceCost()
    guided = run.guidance_scale > 0.0
    provider = make_provider(
        run.provider, architect, schedule, architect_stats, engineer, clean,
        load, run.clip_denoised, cost,
    ) if guided else None

    chunks: list[np.ndarray] = []
    trace: list[dict] = []
    guided_steps = [i for i in range(run.steps) if i % run.guide_every == 0] if guided else []
    picks = np.unique(np.linspace(0, len(guided_steps) - 1, min(SNAPSHOTS, len(guided_steps))).round())
    snapshot_steps = {guided_steps[int(i)] for i in picks} if guided_steps else set()
    recorded: list[dict[str, np.ndarray]] = []
    recorder_meta: dict = {}
    started = time.perf_counter()
    for begin in range(0, noise.shape[0], batch_size):
        batch = noise[begin:begin + batch_size]
        # Each chunk re-seeds so the ancestral noise stream depends only on the
        # position in the pool, never on how the pool was chunked.
        generator = torch.Generator(device=device).manual_seed(run.seed + begin)
        last: dict = {}
        recorder = GuidanceRecorder(snapshot_steps, run.grad_clip) if guided else None

        def field(state: torch.Tensor, timestep: int, index: int, total: int) -> torch.Tensor:
            with torch.no_grad():
                velocity = architect(
                    state, torch.full((state.shape[0],), float(timestep),
                                      device=device, dtype=torch.float32)
                ).sample
            cost.architect += 1
            if not guided or index % run.guide_every:
                return velocity
            gradient, mf = provider(state, timestep)
            weight = bell_weight(index, total, run.guide_w_max, run.bell_peak, run.bell_width)
            correction = torch.clamp(weight * run.guidance_scale * gradient,
                                     -run.grad_clip, run.grad_clip)
            _, sigma = schedule.alpha_sigma(timestep)
            recorder.add(index, timestep, weight, correction, mf, float(sigma))
            last.update(t=timestep, mf=float(mf.mean()), weight=weight,
                        grad=float(correction.flatten(1).norm(dim=1).mean()))
            return velocity + sigma.to(velocity.dtype) * correction

        def record(index: int, timestep: int, state: torch.Tensor) -> None:
            if begin == 0 and last:
                trace.append(dict(last))

        states = sample(field, batch, schedule, run.steps, eta=run.eta,
                        clip_denoised=run.clip_denoised, generator=generator,
                        step_callback=record)
        chunks.append(denormalize_heights(states, architect_stats))
        if recorder is not None:
            recorded.append(recorder.result(int(states.shape[-1])))
            recorder_meta = {"t": np.array(recorder.timesteps), "weight": np.array(recorder.weights),
                             "snapshot_steps": np.array(sorted(snapshot_steps))}
    elapsed = time.perf_counter() - started
    guidance = ({key: np.concatenate([r[key] for r in recorded], axis=0) for key in recorded[0]}
                | recorder_meta) if recorded else None

    heights = np.concatenate(chunks, axis=0)
    metadata = run.to_dict() | {
        "samples": int(heights.shape[0]),
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(heights.shape[0], 1),
        "architect_evaluations": cost.architect,
        "surrogate_evaluations": cost.surrogate,
        "batch_size": batch_size,
        "seed": run.seed,
        "trace": trace,
    }
    return heights, metadata, guidance


def main() -> None:
    args = parse_args()
    runs = matrix.build(args.blocks)
    if args.only:
        runs = [r for r in runs if any(token in r.run_id for token in args.only)]
    print(matrix.summarise(runs), flush=True)

    device = resolve_device(args.device)
    architect_path = args.models_dir / "architect_solid.pt"
    engineer_path = args.models_dir / "engineer_solid.pt"
    print(f"device={device} torch={torch.__version__} host={platform.node()}", flush=True)

    architect, schedule, architect_stats = load_architect(architect_path, device)
    surrogates = {"pbunet": load_surrogate("pbunet", engineer_path, device)}
    for key, filename in (("clean", "clean_solid.pt"), ("hybrid", "engineer_hybrid.pt")):
        candidate = args.models_dir / filename
        if candidate.exists():
            surrogates[key] = load_surrogate(key, candidate, device)
            print(f"loaded optional surrogate {key} from {candidate.name}", flush=True)

    needed = {r.engineer for r in runs} | {
        "clean" for r in runs if r.provider in ("tweedie_clean", "naive_clean")
    }
    missing = sorted(n for n in needed if n not in surrogates and n != "pbunet")
    if missing:
        print(f"WARNING: skipping runs that need {missing}", flush=True)
        runs = [r for r in runs
                if r.engineer in surrogates
                and not (r.provider in ("tweedie_clean", "naive_clean") and "clean" not in surrogates)]
        print(f"{len(runs)} runs remain", flush=True)

    load = normalize_load(args.conditioning, surrogates["pbunet"].stats, device)
    pools: dict[int, torch.Tensor] = {}

    def noise_for(seed: int) -> torch.Tensor:
        if seed not in pools:
            pools[seed] = initial_noise(args.samples, int(architect.config.sample_size), device, seed)
        return pools[seed]

    noise = noise_for(SEED)

    if args.dry_run:
        # Preflight: everything is loaded, nothing is generated. Run this on the
        # GPU box before committing hours to a sweep.
        print(f"\nsurrogates available : {sorted(surrogates)}")
        print(f"conditioning         : {args.conditioning.name} "
              f"-> fz in [{float(load.min()):.3f}, {float(load.max()):.3f}] normalised")
        print(f"initial noise pool   : {tuple(noise.shape)}")
        print(f"architect z domain   : [{architect_stats['z_min']}, {architect_stats['z_max']}]")
        for key, surrogate in sorted(surrogates.items()):
            parameters = sum(p.numel() for p in surrogate.model.parameters())
            print(f"  {key:<8} {parameters/1e6:7.1f} M  z domain "
                  f"[{surrogate.stats['z_min']}, {surrogate.stats['z_max']}]")
        print(f"\n{len(runs)} runs would be generated:")
        for run in runs:
            print(f"  {run.slug}  {run.run_id}")
        return

    samples_dir = args.output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    done = {entry["run_id"] for entry in manifest}

    for position, run in enumerate(runs, start=1):
        if run.run_id in done:
            print(f"[{position}/{len(runs)}] skip {run.run_id}", flush=True)
            continue
        heights, metadata, guidance = run_one(run, noise_for(run.seed), architect, schedule, architect_stats,
                                              surrogates, load, args.batch_size, device)
        target = samples_dir / f"{run.slug}.npz"
        np.savez_compressed(target, z=heights)
        metadata["file"] = target.name
        if guidance is not None:
            guidance_target = samples_dir / f"{run.slug}_guidance.npz"
            np.savez_compressed(guidance_target, **guidance)
            metadata["guidance_file"] = guidance_target.name
        manifest.append(metadata)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{position}/{len(runs)}] {run.run_id} | {metadata['seconds']:.1f}s "
              f"| arch {metadata['architect_evaluations']} "
              f"| surr {metadata['surrogate_evaluations']}", flush=True)

    print(f"\nwrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
