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

SEED = 20260922


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="holds architect_solid.pt, engineer_solid.pt and optionally clean_solid.pt")
    parser.add_argument("--conditioning", type=Path, required=True,
                        help="npz with the fz field, e.g. shell_2600.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", nargs="*", default=None, help=f"subset of {sorted(matrix.BLOCKS)}")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initial_noise(samples: int, size: int, device: torch.device) -> torch.Tensor:
    """One shared pool of initial states, so every run is paired."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
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
    started = time.perf_counter()
    for begin in range(0, noise.shape[0], batch_size):
        batch = noise[begin:begin + batch_size]
        # Each chunk re-seeds so the ancestral noise stream depends only on the
        # position in the pool, never on how the pool was chunked.
        generator = torch.Generator(device=device).manual_seed(SEED + begin)
        last: dict = {}

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
    elapsed = time.perf_counter() - started

    heights = np.concatenate(chunks, axis=0)
    metadata = run.to_dict() | {
        "samples": int(heights.shape[0]),
        "seconds": elapsed,
        "seconds_per_sample": elapsed / max(heights.shape[0], 1),
        "architect_evaluations": cost.architect,
        "surrogate_evaluations": cost.surrogate,
        "batch_size": batch_size,
        "seed": SEED,
        "trace": trace,
    }
    return heights, metadata


def main() -> None:
    args = parse_args()
    runs = matrix.build(args.blocks)
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
    noise = initial_noise(args.samples, int(architect.config.sample_size), device)

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
        heights, metadata = run_one(run, noise, architect, schedule, architect_stats,
                                    surrogates, load, args.batch_size, device)
        target = samples_dir / f"{run.slug}.npz"
        np.savez_compressed(target, z=heights)
        metadata["file"] = target.name
        manifest.append(metadata)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{position}/{len(runs)}] {run.run_id} | {metadata['seconds']:.1f}s "
              f"| arch {metadata['architect_evaluations']} "
              f"| surr {metadata['surrogate_evaluations']}", flush=True)

    print(f"\nwrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
