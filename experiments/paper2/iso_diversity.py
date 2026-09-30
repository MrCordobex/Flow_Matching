"""Quality at matched diversity: the best FEM Membrane Factor each surrogate reaches
while the generated set keeps at least a given diversity. Exploratory, not in
PREREGISTRO.md.

    uv run python experiments/paper2/iso_diversity.py results/budget --vae <solid_vae/best.pt>

For every evaluator family and step budget K the runs of b6/b6r/b6u/b7 form a
curve over gamma (0, 10, 25, 50, 100, 250), with b6u as the shared unguided
point. Along that polyline, piecewise linear in (diversity, mf), the value
reported at a diversity level d* is the largest mf among points whose diversity
is >= d*: the upper envelope, well defined even when the curve is not monotone.

Intervals are paired bootstrap over sample indices: one resample of the 100
pool positions is applied to every run, and both the mean FEM mf and the mean
pairwise diversity are recomputed on it. Diversity is measured twice, in the
2-D latent of the paper-1 VAE and as model-free mean pairwise L2 in height
space, so a conclusion that holds in only one of them is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STEPS = (10, 20, 100)
BLOCKS = ("b6", "b6r", "b6u", "b7")
LEVELS = 9
RESAMPLES = 2_000
MF_KEY = "mf_resultants_area_mean"
RNG = np.random.default_rng(20260930)
PROVIDER = {"noise_aware": "NA", "tweedie_clean": "TC", "tweedie_self": "TS"}
BASELINE = "TC-pbunet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--vae", type=Path, required=True)
    return parser.parse_args()


def family(entry: dict) -> str:
    return f"{PROVIDER[entry['provider']]}-{entry['engineer']}"


def encode(heights: np.ndarray, checkpoint: dict) -> np.ndarray:
    import torch

    from paper2.vae import load_vae

    stats = checkpoint.get("normalization_stats", {})
    low = float(stats.get("z_min", 0.0))
    high = float(stats.get("z_max", 7.850908857450063))
    z = heights[:, 0] if heights.ndim == 4 else heights
    model = load_vae(checkpoint)
    with torch.no_grad():
        return model.encode(torch.from_numpy(2.0 * (z - low) / (high - low) - 1.0).unsqueeze(1).float()).numpy()


def pairwise(points: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))


def fem(results: Path, slug: str, n: int) -> np.ndarray:
    mf = np.full(n, np.nan)
    for row in csv.DictReader((results / "kratos" / slug / "metrics.csv").open(encoding="utf-8")):
        if row["status"] == "ok":
            mf[int(row["sample_index"])] = float(row[MF_KEY])
    return mf


def load(results: Path, vae: Path) -> list[dict]:
    import torch

    checkpoint = torch.load(vae, map_location="cpu", weights_only=False)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    runs = []
    for entry in manifest:
        if entry["block"] not in BLOCKS or entry["steps"] not in STEPS:
            continue
        heights = np.load(results / "samples" / entry["file"])["z"]
        flat = heights.reshape(len(heights), -1)
        runs.append({
            "family": family(entry),
            "steps": entry["steps"],
            "gamma": 0.0 if entry["block"] == "b6u" else float(entry["guidance_scale"]),
            "mf": fem(results, entry["slug"], len(heights)),
            "latent": pairwise(encode(heights, checkpoint)),
            "height": pairwise(flat) / np.sqrt(flat.shape[1]),
        })
    return runs


def stats(run: dict, index: np.ndarray, metric: str) -> tuple[float, float]:
    distances = run[metric][np.ix_(index, index)]
    off = index[:, None] != index[None, :]  # a resampled duplicate is not a pair
    return float(np.nanmean(run["mf"][index])), float(distances[off].mean())


def envelope(points: list[tuple[float, float]], levels: np.ndarray) -> np.ndarray:
    """Best mf with diversity >= level, along the gamma-ordered polyline."""
    mf = np.array([p[0] for p in points])
    div = np.array([p[1] for p in points])
    fine = np.linspace(0.0, 1.0, 50)
    seg_mf = np.concatenate([mf[i] + fine * (mf[i + 1] - mf[i]) for i in range(len(mf) - 1)] or [mf])
    seg_div = np.concatenate([div[i] + fine * (div[i + 1] - div[i]) for i in range(len(div) - 1)] or [div])
    out = np.full(len(levels), np.nan)
    for j, level in enumerate(levels):
        ok = seg_div >= level
        if ok.any():
            out[j] = seg_mf[ok].max()
    return out


def curves(runs: list[dict], steps: int, index: np.ndarray, metric: str) -> dict[str, list]:
    unguided = [r for r in runs if r["steps"] == steps and r["gamma"] == 0.0]
    out = {}
    for name in sorted({r["family"] for r in runs if r["gamma"] > 0}):
        members = sorted((r for r in runs if r["steps"] == steps and r["family"] == name and r["gamma"] > 0),
                         key=lambda r: r["gamma"])
        out[name] = [stats(r, index, metric) for r in unguided + members]
    return out


def main() -> None:
    args = parse_args()
    runs = load(args.results, args.vae)
    identity = np.arange(len(runs[0]["mf"]))
    report: dict = {}
    for metric in ("latent", "height"):
        for steps in STEPS:
            point = curves(runs, steps, identity, metric)
            unguided_div = point[BASELINE][0][1]
            # Levels stay inside the range every NA family and TC cover, with a
            # margin, so a family is never judged at a diversity it cannot reach.
            # TS collapses little and would otherwise set the grid; it is still
            # reported where feasible.
            ranged = [c for name, c in point.items() if not name.startswith("TS")]
            floor = max(min(d for _, d in c) for c in ranged)
            ceiling = min(max(d for _, d in c) for c in ranged)
            margin = 0.05 * (ceiling - floor)
            levels = np.linspace(ceiling - margin, floor + margin, LEVELS)
            base = {name: envelope(c, levels) for name, c in point.items()}
            boot = {name: np.empty((RESAMPLES, LEVELS)) for name in point}
            for b in range(RESAMPLES):
                index = RNG.integers(0, len(identity), len(identity))
                for name, c in curves(runs, steps, index, metric).items():
                    boot[name][b] = envelope(c, levels)
            cell = {"unguided_diversity": unguided_div, "levels": levels.tolist(), "families": {}}
            for name in point:
                entry = {"mf": base[name].tolist(), "curve": point[name]}
                if name != BASELINE:
                    diff = boot[name] - boot[BASELINE]
                    entry["diff_vs_tc"] = (base[name] - base[BASELINE]).tolist()
                    entry["ci_low"] = np.nanpercentile(diff, 2.5, axis=0).tolist()
                    entry["ci_high"] = np.nanpercentile(diff, 97.5, axis=0).tolist()
                    entry["feasible_share"] = np.isfinite(diff).mean(0).tolist()
                cell["families"][name] = entry
            report[f"{metric}_K{steps}"] = cell
            print_cell(metric, steps, cell)
    output = args.results / "iso_diversity.json"
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {output}")


def print_cell(metric: str, steps: int, cell: dict) -> None:
    print(f"\n=== diversity: {metric}   K={steps}   unguided d={cell['unguided_diversity']:.3f}")
    print(f"{'d*':>7} " + " ".join(f"{name:>24}" for name in cell["families"]))
    for j, level in enumerate(cell["levels"]):
        parts = []
        for name, entry in cell["families"].items():
            mf = entry["mf"][j]
            if "diff_vs_tc" in entry and np.isfinite(mf) and np.isfinite(entry["diff_vs_tc"][j]):
                share = entry["feasible_share"][j]
                flag = "" if share > 0.95 else "?"
                parts.append(f"{mf:.3f} {entry['diff_vs_tc'][j]:+.3f}[{entry['ci_low'][j]:+.2f},{entry['ci_high'][j]:+.2f}]{flag}")
            else:
                parts.append(f"{mf:.3f}" if np.isfinite(mf) else "—")
        print(f"{level:7.3f} " + " ".join(f"{p:>24}" for p in parts))


if __name__ == "__main__":
    main()
