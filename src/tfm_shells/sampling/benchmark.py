"""Paired numerical convergence experiment for the flow sampler."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from tfm_shells.config import load_config, resolve_project_path
from tfm_shells.sampling.guided import denormalize_samples, evaluate_clean_mf, generate_samples, load_sampling_context
from tfm_shells.training.common import resolve_device, seed_everything
from tfm_shells.utils.io import save_json


def benchmark_steps(config_path: Path, steps: list[int], reference_steps: int, output: Path) -> list[dict]:
    config = load_config(config_path)
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    seed_everything(int(config["seed"]))
    context = load_sampling_context(config, device)
    batch_size = int(config["conditioning"]["batch_size"])
    size = int(context.architect.config.sample_size)
    initial = torch.randn((batch_size, 1, size, size), device=device)
    solver = str(config["sampling"].get("solver", "euler"))
    counts = sorted(set([reference_steps, *steps]))
    if any(count < 1 for count in counts):
        raise ValueError("All step counts must be positive")

    results = {}
    for count in counts:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        states, _ = generate_samples(context, config, initial.clone(), count, solver)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        z = denormalize_samples(context, states)
        mf = evaluate_clean_mf(context, states)
        results[count] = (z, mf, elapsed)
        print(f"steps={count:4d} seconds={elapsed:.2f} mean_MF={mf.mean():.4f}", flush=True)

    reference_z, reference_mf, _ = results[reference_steps]
    output_path = resolve_project_path(config, output)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for count in counts:
        z, mf, elapsed = results[count]
        difference = z - reference_z
        row = {
            "steps": count,
            "solver": solver,
            "guided": float(config["sampling"]["guidance_scale"]) > 0,
            "seconds": elapsed,
            "model_evaluations": count * (2 if solver == "heun" else 1),
            "mf_mean": float(mf.mean()),
            "mf_std": float(mf.std()),
            "p_mf_gt_090": float((mf > 0.90).mean()),
            "mf_mean_delta_from_reference": float(mf.mean() - reference_mf.mean()),
            "paired_mf_mae_from_reference": float(np.abs(mf - reference_mf).mean()),
            "paired_z_mae_m": float(np.abs(difference).mean()),
            "paired_z_rmse_m": float(np.sqrt(np.square(difference).mean())),
        }
        rows.append(row)
    with (output_path / "step_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        {"reference_steps": reference_steps, "solver": solver, "seed": int(config["seed"]),
         "batch_size": batch_size, "guidance_scale": float(config["sampling"]["guidance_scale"]),
         "comparison": "same initial Gaussian noise, same trained checkpoints, same solver and load"},
        output_path / "metadata.json",
    )
    return rows
