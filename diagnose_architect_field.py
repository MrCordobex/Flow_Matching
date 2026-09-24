"""Measure VP Architect errors by noise level and spatial scale on held-out shells.

Example:
    uv run python diagnose_architect_field.py --checkpoint best.pt \
        --data-zip solid_npz.zip --output artifacts/architect_field_diagnostic

The validation split is reconstructed from the training config (seed 42, 20%).
No surrogate or guidance is used. All derivatives are finite differences per pixel.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from tfm_shells.cosine_flow import FLOW_METHOD, CosineFlowPath
from tfm_shells.models.factory import build_unet
from tfm_shells.vp_diffusion import CosineVPSchedule, vp_model_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/architect_field_diagnostic"))
    parser.add_argument("--samples", type=int, default=24, help="Held-out geometries to evaluate")
    parser.add_argument("--repeats", type=int, default=1, help="Independent noise draws per geometry")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Training split and evaluation seed")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--times", type=float, nargs="+",
                        default=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.99])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-split-check", action="store_true",
                        help="Skip verifying validation MF against the checkpoint (faster)")
    return parser.parse_args()


def read_npz(zip_file: zipfile.ZipFile, name: str, key: str) -> np.ndarray:
    with np.load(io.BytesIO(zip_file.read(name)), allow_pickle=False) as data:
        return np.asarray(data[key]).copy()


def spectrum_error(error: torch.Tensor, prediction: torch.Tensor,
                   truth: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image RMS contribution of each radial Fourier band, in height units."""
    height, width = error.shape[-2:]
    fy = torch.fft.fftfreq(height, device=error.device)[:, None]
    fx = torch.fft.fftfreq(width, device=error.device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    transform = torch.fft.fft2(error.float(), norm="ortho")
    power = transform.abs().square()
    result = {}
    for key, mask in (
        ("low", radius < 0.125),
        ("mid", (radius >= 0.125) & (radius < 0.25)),
        ("high", radius >= 0.25),
    ):
        # With orthonormal FFT, sum(power) / (H*W) = spatial MSE.
        result[f"{key}_band_rmse"] = (power[..., mask].sum(dim=-1) / (height * width)).sqrt().squeeze(1)
    mask = radius >= 0.25
    pred_power = torch.fft.fft2(prediction.float(), norm="ortho").abs().square()
    real_power = torch.fft.fft2(truth.float(), norm="ortho").abs().square()
    result["high_band_power_ratio"] = (
        (pred_power[..., mask].sum(dim=-1) / real_power[..., mask].sum(dim=-1).clamp_min(1e-12))
        .squeeze(1)
    )
    return result


def sample_metrics(prediction: torch.Tensor, truth: torch.Tensor,
                   predicted_velocity: torch.Tensor, true_velocity: torch.Tensor) -> dict[str, torch.Tensor]:
    error = prediction - truth
    dx = error[..., 1:] - error[..., :-1]
    dy = error[..., 1:, :] - error[..., :-1, :]
    lap = (error[..., :-2, 1:-1] + error[..., 2:, 1:-1]
           + error[..., 1:-1, :-2] + error[..., 1:-1, 2:]
           - 4 * error[..., 1:-1, 1:-1])
    result = {
        "velocity_rmse": (predicted_velocity - true_velocity).square().mean(dim=(1, 2, 3)).sqrt(),
        "height_rmse": error.square().mean(dim=(1, 2, 3)).sqrt(),
        "slope_rmse": ((dx.square().mean(dim=(1, 2, 3))
                         + dy.square().mean(dim=(1, 2, 3))) / 2).sqrt(),
        "curvature_rmse": lap.square().mean(dim=(1, 2, 3)).sqrt(),
    }
    result.update(spectrum_error(error, prediction, truth))
    return result


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.repeats < 1 or args.batch_size < 1:
        raise ValueError("samples, repeats and batch-size must be positive")
    if any(not 0 < t < 1 for t in args.times):
        raise ValueError("every evaluation time must satisfy 0 < t < 1")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    diffusion = checkpoint["diffusion_config"]
    if diffusion["method"] not in {"cosine_vp_v", FLOW_METHOD} or float(diffusion["time_scale"]) != 999:
        raise ValueError("checkpoint is not a cosine Architect (v prediction or flow matching)")
    # Flow checkpoints emit dx/dt; every metric below is defined on dx/dphi.
    net_to_v = (1.0 / CosineFlowPath(float(diffusion["cosine_s"])).rate
                if diffusion["method"] == FLOW_METHOD else 1.0)
    stats = checkpoint["normalization_stats"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = build_unet(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint["optimizer_state_dict"]
    model.to(device).eval()
    schedule = CosineVPSchedule(float(diffusion["cosine_s"]))
    z_scale = (float(stats["z_max"]) - float(stats["z_min"])) / 2.0

    with zipfile.ZipFile(args.data_zip) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.endswith(".npz") and "shell_hole" not in name)
        _, validation = train_test_split(names, test_size=args.val_ratio,
                                         random_state=args.seed, shuffle=True)
        checkpoint_summary = checkpoint["dataset_summary"]
        expected_count = int(checkpoint_summary["all_filtered"]["count"])
        if len(names) != expected_count:
            raise ValueError(f"ZIP has {len(names)} solids, checkpoint trained on {expected_count}")
        if not args.skip_split_check:
            mf_mean = float(np.mean([read_npz(archive, name, "mf").mean()
                                     for name in validation]))
            expected_mf = float(checkpoint_summary["val"]["mf_mean"])
            if not np.isclose(mf_mean, expected_mf, atol=1e-6):
                raise ValueError(f"Validation split mismatch: MF {mf_mean:.8f}, checkpoint {expected_mf:.8f}")
        generator = np.random.default_rng(args.seed)
        chosen = sorted(generator.choice(validation, size=min(args.samples, len(validation)),
                                         replace=False).tolist())
        clean = np.stack([read_npz(archive, name, "z").astype(np.float32)
                          for name in chosen])
    clean = torch.from_numpy(2.0 * (clean - float(stats["z_min"])) /
                             (float(stats["z_max"]) - float(stats["z_min"])) - 1.0)
    height = int(checkpoint["model_config"]["sample_size"])
    if tuple(clean.shape[1:]) != (1, height, height):
        raise ValueError(f"unexpected z shape: {tuple(clean.shape)}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    noise_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    with torch.inference_mode():
        for repeat in range(args.repeats):
            noise = torch.randn(clean.shape, generator=noise_gen)
            for t in sorted(args.times):
                alpha, sigma = schedule.alpha_sigma(torch.tensor(t, dtype=torch.float32))
                for start in range(0, len(chosen), args.batch_size):
                    stop = min(start + args.batch_size, len(chosen))
                    x0 = clean[start:stop].to(device)
                    eps = noise[start:stop].to(device)
                    state = alpha * x0 + sigma * eps
                    target = alpha * eps - sigma * x0
                    predicted = model(state, vp_model_time(t, len(x0), device)).sample * net_to_v
                    estimated_z = (alpha * state - sigma * predicted) * z_scale
                    real_z = x0 * z_scale
                    measured = sample_metrics(estimated_z, real_z, predicted, target)
                    for i, name in enumerate(chosen[start:stop]):
                        row: dict[str, float | int | str] = {"name": name, "repeat": repeat, "t": t,
                                                              "sigma": float(sigma)}
                        row.update({key: float(value[i].cpu()) for key, value in measured.items()})
                        rows.append(row)
                print(f"time {t:.3f} | {len(chosen)} validation samples | repeat {repeat + 1}/{args.repeats}", flush=True)

    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [key for key in rows[0] if key not in {"name", "repeat", "t", "sigma"}]
    aggregate = []
    for t in sorted(args.times):
        records = [row for row in rows if row["t"] == t]
        record = {"t": t, "sigma": float(records[0]["sigma"]), "n": len(records)}
        record.update({key: float(np.mean([row[key] for row in records])) for key in metric_names})
        aggregate.append(record)
    with (args.output / "by_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    t_values = [record["t"] for record in aggregate]
    for ax, keys, ylabel in (
        (axs[0, 0], ["velocity_rmse", "height_rmse"], "RMSE (velocity normalized / height units)"),
        (axs[0, 1], ["slope_rmse", "curvature_rmse"], "RMSE (height units / pixel or pixel²)"),
        (axs[1, 0], ["low_band_rmse", "mid_band_rmse", "high_band_rmse"], "Height error by spatial band"),
        (axs[1, 1], ["high_band_power_ratio"], "Predicted / true high-frequency power"),
    ):
        for key in keys:
            ax.plot(t_values, [record[key] for record in aggregate], marker="o", label=key)
        ax.set(xlabel="VP time t (0: clean, 1: noise)", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Architect epoch {checkpoint['epoch']}, validation set, {len(chosen)} shells")
    fig.tight_layout()
    fig.savefig(args.output / "by_time.png", dpi=160)
    plt.close(fig)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint["epoch"],
                "architect_method": diffusion["method"],
                "velocity_units": "dx/dphi (flow checkpoints rescaled by 1/rate)",
                "data_zip": str(args.data_zip.resolve()), "split_seed": args.seed,
                "split_val_ratio": args.val_ratio, "validation_count": len(validation),
                "samples": len(chosen), "repeats": args.repeats, "sample_names": chosen,
                "z_scale": z_scale, "slope_units": "height unit / pixel",
                "curvature_units": "height unit / pixel squared",
                "bands_cycles_per_pixel": {"low": "[0,0.125)", "mid": "[0.125,0.25)",
                                            "high": "[0.25,0.707]"},
                "interpretation": "Teacher-forced noisy states; high t includes intrinsic uncertainty, not only model error"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nt     v_RMSE   z_RMSE   slope_RMSE   curvature_RMSE   high_band_RMSE")
    for row in aggregate:
        print(f"{row['t']:.2f}  {row['velocity_rmse']:.4f}   {row['height_rmse']:.4f}    "
              f"{row['slope_rmse']:.4f}       {row['curvature_rmse']:.4f}           "
              f"{row['high_band_rmse']:.4f}")
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
