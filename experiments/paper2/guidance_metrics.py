"""Per-sample guidance metrics defined in PREREGISTRO.md, section 5.

Reads the `*_guidance.npz` files generate.py writes for guided runs, pairs each
run with its unguided twin (same K, eta, clip and seed, hence the same initial
and ancestral noise) and joins the Kratos verdicts.

    uv run python experiments/paper2/guidance_metrics.py results/budget

Writes guidance_samples.csv (one row per sample) and guidance_runs.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

MF_KEY = "mf_resultants_area_mean"
EARLY_T = 500  # anticipation is measured on the high-noise half of the trajectory


def load_fem(results: Path, entry: dict) -> np.ndarray:
    """Kratos MF per sample, NaN where the solver failed or has not run yet."""
    values = np.full(entry["samples"], np.nan)
    path = results / "kratos" / entry["slug"] / "metrics.csv"
    if path.exists():
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if row["status"] == "ok":
                values[int(row["sample_index"])] = float(row[MF_KEY])
    return values


def evaluator(entry: dict) -> str:
    if entry["guidance_scale"] == 0.0:
        return "unguided"
    if entry["engineer"] == "hybrid":
        return "HY"
    return {"noise_aware": "NA", "tweedie_clean": "TC", "tweedie_self": "TS",
            "naive_clean": "NC"}[entry["provider"]]


def twin_key(entry: dict) -> tuple:
    return entry["steps"], entry["eta"], entry["clip_denoised"], entry.get("seed", 20260922)


def per_sample(results: Path, entry: dict, twin: dict | None) -> list[dict]:
    guidance = np.load(results / "samples" / entry["guidance_file"])
    z = np.load(results / "samples" / entry["file"])["z"]
    z = z[:, 0] if z.ndim == 4 else z
    fem = load_fem(results, entry)

    path = guidance["path"]
    straightness = np.linalg.norm(guidance["cumulative"].reshape(len(path), -1), axis=1) / np.maximum(path, 1e-12)
    coherence = np.nanmean(guidance["cos_prev"], axis=1) if guidance["cos_prev"].shape[1] > 1 \
        else np.full(len(path), np.nan)
    t = guidance["t"]
    norm_d = guidance["norm_d"]
    early_share = norm_d[:, t >= EARLY_T].sum(1) / np.maximum(path, 1e-12)

    anticipation = np.full(len(path), np.nan)
    efficiency = np.full(len(path), np.nan)
    fem_twin = np.full(len(path), np.nan)
    if twin is not None:
        z_twin = np.load(results / "samples" / twin["file"])["z"]
        z_twin = z_twin[:, 0] if z_twin.ndim == 4 else z_twin
        shift = (z - z_twin).reshape(len(path), -1)
        # snapshot_steps are step indices; map them to timesteps through the guided-step order
        guided_index = {int(step): position for position, step in enumerate(_guided_steps(entry))}
        snap_timesteps = np.array([t[guided_index[int(s)]] for s in guidance["snapshot_steps"]])
        early = snap_timesteps >= EARLY_T
        if early.any():
            snaps = guidance["snapshots"][:, early].astype(np.float32).reshape(len(path), early.sum(), -1)
            dots = (snaps * shift[:, None, :]).sum(-1)
            norms = np.linalg.norm(snaps, axis=-1) * np.linalg.norm(shift, axis=-1)[:, None]
            anticipation = np.nanmean(dots / np.maximum(norms, 1e-12), axis=1)
        fem_twin = load_fem(results, twin)
        efficiency = (fem - fem_twin) / np.maximum(path, 1e-12)

    rows = []
    for i in range(len(path)):
        rows.append({
            "run_id": entry["run_id"], "block": entry["block"], "evaluator": evaluator(entry),
            "steps": entry["steps"], "gamma": entry["guidance_scale"], "seed": entry.get("seed", 20260922),
            "sample": i, "fem_mf": fem[i], "fem_mf_twin": fem_twin[i],
            "push": path[i], "straightness": straightness[i], "coherence": coherence[i],
            "anticipation": anticipation[i], "efficiency": efficiency[i], "early_share": early_share[i],
            "clip_frac": float(np.mean(guidance["clip_frac"][i])),
            "mf_pred_final": float(guidance["mf_pred"][i, -1]),
        })
    return rows


def _guided_steps(entry: dict) -> list[int]:
    return [i for i in range(entry["steps"]) if i % entry["guide_every"] == 0]


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/budget")
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    twins: dict[tuple, dict] = {}
    for entry in manifest:  # prefer the dedicated b6u twins, fall back to any unguided run
        if entry["guidance_scale"] == 0.0 and (twin_key(entry) not in twins or entry["block"] == "b6u"):
            twins[twin_key(entry)] = entry

    rows: list[dict] = []
    for entry in manifest:
        if "guidance_file" not in entry:
            continue
        rows.extend(per_sample(results, entry, twins.get(twin_key(entry))))
    if not rows:
        print("no instrumented runs in the manifest yet")
        return

    with (results / "guidance_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for run_id in dict.fromkeys(r["run_id"] for r in rows):
        chosen = [r for r in rows if r["run_id"] == run_id]
        first = chosen[0]
        record = {k: first[k] for k in ("run_id", "block", "evaluator", "steps", "gamma", "seed")}
        for key in ("fem_mf", "push", "straightness", "coherence", "anticipation", "efficiency",
                    "early_share", "clip_frac", "mf_pred_final"):
            values = np.array([r[key] for r in chosen], dtype=float)
            record[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        record["fem_n"] = int(np.isfinite([r["fem_mf"] for r in chosen]).sum())
        summary.append(record)
    with (results / "guidance_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"{'run':<52}{'fem':>7}{'push':>8}{'straight':>9}{'coher':>7}{'antic':>7}{'effic':>8}{'early':>7}")
    for r in sorted(summary, key=lambda r: (r["block"], r["evaluator"], r["steps"], r["gamma"])):
        print(f"{r['run_id'][:51]:<52}{r['fem_mf']:>7.3f}{r['push']:>8.3f}{r['straightness']:>9.3f}"
              f"{r['coherence']:>7.3f}{r['anticipation']:>7.3f}{r['efficiency']:>8.3f}{r['early_share']:>7.2f}")
    print(f"\nwrote {results / 'guidance_samples.csv'} and guidance_runs.csv")


if __name__ == "__main__":
    main()
