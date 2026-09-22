"""Join generation cost with FEM verdicts and report the sampling-budget frontier.

Quality is read from Kratos, never from the surrogate that produced the
guidance gradient. `mf_resultants_area_mean` is the metric used, because it is
the one that reproduces the Abaqus convention the dataset was built with.

Diversity uses the auxiliary VAE of the published paper when its checkpoint is
supplied; otherwise the geometric proxies are still reported.

    python -m paper2.analyze --results <dir> [--vae <best.pt>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MF_KEY = "mf_resultants_area_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.90)
    return parser.parse_args()


def roughness(heights: np.ndarray) -> tuple[float, float]:
    """Mean and max |laplacian| of the elevation map, in centimetres."""
    z = heights[:, 0] if heights.ndim == 4 else heights
    lap = (z[:, :-2, 1:-1] + z[:, 2:, 1:-1] + z[:, 1:-1, :-2]
           + z[:, 1:-1, 2:] - 4 * z[:, 1:-1, 1:-1])
    return float(np.abs(lap).mean() * 100), float(np.abs(lap).max() * 100)


def geometric_diversity(heights: np.ndarray) -> dict[str, float]:
    z = heights[:, 0] if heights.ndim == 4 else heights
    peaks = z.reshape(z.shape[0], -1).max(axis=1)
    flat = z.reshape(z.shape[0], -1)
    # Mean pairwise L2 in height space: a model-free stand-in for latent spread.
    distances = np.sqrt(((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1))
    upper = distances[np.triu_indices(len(flat), k=1)]
    return {
        "cv_zmax": float(peaks.std() / max(peaks.mean(), 1e-9)),
        "mean_pairwise_distance": float(upper.mean()) if upper.size else 0.0,
    }


def latent_diversity(heights: np.ndarray, vae_path: Path) -> dict[str, float]:
    """Convex-hull area and mean pairwise distance in the auxiliary VAE latent."""
    import torch
    from scipy.spatial import ConvexHull

    checkpoint = torch.load(vae_path, map_location="cpu", weights_only=False)
    stats = checkpoint.get("normalization_stats", {})
    low = float(stats.get("z_min", 0.0))
    high = float(stats.get("z_max", 7.850908857450063))
    z = heights[:, 0] if heights.ndim == 4 else heights
    normalized = 2.0 * (z - low) / (high - low) - 1.0

    from paper2.vae import load_vae  # kept separate: the VAE class lives with the paper code

    model = load_vae(checkpoint)
    with torch.no_grad():
        latents = model.encode(torch.from_numpy(normalized).unsqueeze(1).float()).numpy()
    result = {"latent_dim": int(latents.shape[1])}
    pairwise = np.sqrt(((latents[:, None, :] - latents[None, :, :]) ** 2).sum(-1))
    upper = pairwise[np.triu_indices(len(latents), k=1)]
    result["latent_mean_distance"] = float(upper.mean()) if upper.size else 0.0
    if latents.shape[1] >= 2 and len(latents) > latents.shape[1]:
        try:
            result["latent_hull_area"] = float(ConvexHull(latents[:, :2]).volume)
        except Exception:
            result["latent_hull_area"] = float("nan")
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest:
        run_id = entry["run_id"]
        metrics_path = args.results / "kratos" / entry["slug"] / "metrics.csv"
        row = {k: v for k, v in entry.items() if k != "trace"}
        heights = np.load(args.results / "samples" / entry["file"])["z"]
        mean_r, max_r = roughness(heights)
        row |= {"roughness_mean_cm": mean_r, "roughness_max_cm": max_r}
        row |= geometric_diversity(heights)
        if args.vae and args.vae.exists():
            try:
                row |= latent_diversity(heights, args.vae)
            except Exception as error:  # a missing VAE class must not sink the table
                row["latent_error"] = str(error)[:120]

        if metrics_path.exists():
            records = [r for r in csv.DictReader(metrics_path.open(encoding="utf-8"))
                       if r["status"] == "ok"]
            values = np.array([float(r[MF_KEY]) for r in records]) if records else np.array([])
            row |= {
                "fem_n": len(records),
                "fem_failed": sum(1 for r in csv.DictReader(metrics_path.open(encoding="utf-8"))
                                  if r["status"] != "ok"),
                "fem_mf_mean": float(values.mean()) if values.size else float("nan"),
                "fem_mf_std": float(values.std()) if values.size else float("nan"),
                "fem_p_above": float((values > args.threshold).mean()) if values.size else float("nan"),
            }
        else:
            row |= {"fem_n": 0, "fem_mf_mean": float("nan")}
        rows.append(row)

    if not rows:
        print("nothing to analyse")
        return
    fields = sorted({key for row in rows for key in row})
    output = args.results / "frontier.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'run':<58}{'NFE_s':>7}{'s':>8}{'FEM mf':>9}{'P>thr':>7}{'rough':>8}{'roughmx':>9}")
    for row in sorted(rows, key=lambda r: (r["block"], r["provider"], r["steps"], r["eta"])):
        print(f"{row['run_id'][:57]:<58}{row.get('surrogate_evaluations', 0):>7}"
              f"{row.get('seconds', 0):>8.0f}{row.get('fem_mf_mean', float('nan')):>9.4f}"
              f"{row.get('fem_p_above', float('nan')):>7.2f}"
              f"{row.get('roughness_mean_cm', float('nan')):>8.2f}"
              f"{row.get('roughness_max_cm', float('nan')):>9.2f}")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
