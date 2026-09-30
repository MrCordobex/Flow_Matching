"""Comparison table for the literature benchmark, at matched diversity.

    uv run python experiments/paper2/literature_table.py results/literature \
        --vae <solid_vae/best.pt> [--reference ours-pbunet]

Needs the Kratos output of evaluate_all.py in <results>/kratos. Each method is a
curve over its strengths, anchored at the unguided run of the same K. The value
at a diversity level is the best FEM Membrane Factor reachable while the latent
mean pairwise distance stays >= level (upper envelope, see iso_diversity.py).
Levels are fractions of the unguided diversity at that K.

Differences are reference minus method, so a positive number means the
reference (the paper's sampler by default) is better. Intervals are paired
bootstrap over pool positions.

Writes literature_table.csv (long format), literature_table.md and .json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from paper2.iso_diversity import encode, envelope, fem, pairwise, stats

FRACTIONS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
RESAMPLES = 2_000
THRESHOLD = 0.90
RNG = np.random.default_rng(20260930)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--reference", default="ours-pbunet")
    return parser.parse_args()


def load(results: Path, vae: Path) -> list[dict]:
    import torch

    checkpoint = torch.load(vae, map_location="cpu", weights_only=False)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    runs = []
    for entry in manifest:
        if not (results / "kratos" / entry["slug"] / "metrics.csv").exists():
            print(f"  no Kratos output for {entry['run_id']}, left out")
            continue
        heights = np.load(results / "samples" / entry["file"])["z"]
        runs.append({
            "variant": entry["provider"], "steps": int(entry["steps"]),
            "strength": float(entry["guidance_scale"]),
            "mf": fem(results, entry["slug"], len(heights)),
            "latent": pairwise(encode(heights, checkpoint)),
            "seconds_per_sample": float(entry["seconds_per_sample"]),
            "architect_evaluations": entry.get("architect_evaluations"),
            "surrogate_evaluations": entry.get("surrogate_evaluations"),
        })
    return runs


def curve(runs: list[dict], variant: str, steps: int, index: np.ndarray) -> list[tuple[float, float]]:
    base = [r for r in runs if r["variant"] == "unguided" and r["steps"] == steps]
    members = sorted((r for r in runs if r["variant"] == variant and r["steps"] == steps),
                     key=lambda r: r["strength"])
    return [stats(r, index, "latent") for r in base + members]


def main() -> None:
    args = parse_args()
    runs = load(args.results, args.vae)
    variants = sorted({r["variant"] for r in runs if r["variant"] != "unguided"})
    identity = np.arange(len(runs[0]["mf"]))
    rows, report = [], {}
    for steps in sorted({r["steps"] for r in runs}):
        base = [r for r in runs if r["variant"] == "unguided" and r["steps"] == steps]
        if not base:
            print(f"K={steps}: no unguided run, skipped")
            continue
        unguided_div = stats(base[0], identity, "latent")[1]
        levels = np.array(FRACTIONS) * unguided_div
        present = [v for v in variants if any(r["variant"] == v and r["steps"] == steps for r in runs)]
        point = {v: envelope(curve(runs, v, steps, identity), levels) for v in present}
        boot = {v: np.empty((RESAMPLES, len(levels))) for v in present}
        for b in range(RESAMPLES):
            index = RNG.integers(0, len(identity), len(identity))
            for v in present:
                boot[v][b] = envelope(curve(runs, v, steps, index), levels)
        for v in present:
            members = [r for r in runs if r["variant"] == v and r["steps"] == steps]
            strongest = max(members, key=lambda r: r["strength"])
            best = max(members, key=lambda r: np.nanmean(r["mf"]))
            row = {
                "K": steps, "variant": v,
                "seconds_per_sample_max": strongest["seconds_per_sample"],
                "best_mf": float(np.nanmean(best["mf"])),
                "best_p_above": float(np.nanmean(best["mf"] > THRESHOLD)),
                "best_diversity": stats(best, identity, "latent")[1] / unguided_div,
                "fem_failed": int(sum(np.isnan(r["mf"]).sum() for r in members)),
            }
            for j, fraction in enumerate(FRACTIONS):
                row[f"mf@{fraction:.2f}"] = float(point[v][j])
                if v != args.reference and args.reference in point:
                    diff = boot[args.reference][:, j] - boot[v][:, j]
                    row[f"ref-minus@{fraction:.2f}"] = float(point[args.reference][j] - point[v][j])
                    row[f"ci_low@{fraction:.2f}"] = float(np.nanpercentile(diff, 2.5))
                    row[f"ci_high@{fraction:.2f}"] = float(np.nanpercentile(diff, 97.5))
            rows.append(row)
        report[f"K{steps}"] = {"unguided_diversity": unguided_div, "levels": levels.tolist()}

    fields = sorted({k for row in rows for k in row}, key=lambda k: (k not in ("K", "variant"), k))
    with (args.results / "literature_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.results / "literature_table.json").write_text(json.dumps({"rows": rows, "levels": report}, indent=1),
                                                       encoding="utf-8")
    markdown = render(rows, args.reference)
    (args.results / "literature_table.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def render(rows: list[dict], reference: str) -> str:
    shown = (0.95, 0.85, 0.75)
    lines = [f"Best FEM mf at >= x% of the unguided diversity; delta = {reference} minus method, 95% CI.\n"]
    for steps in sorted({r["K"] for r in rows}):
        lines.append(f"\n### K = {steps}\n")
        head = "| method | " + " | ".join(f"mf @{int(f*100)}%" for f in shown) + \
               f" | Δ @85% [CI] | best mf (P>0.9, div) | s/sample | FEM fail |"
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        for row in sorted((r for r in rows if r["K"] == steps), key=lambda r: -np.nan_to_num(r["mf@0.85"])):
            cells = [f"{row[f'mf@{f:.2f}']:.3f}" if np.isfinite(row[f"mf@{f:.2f}"]) else "—" for f in shown]
            if "ref-minus@0.85" in row and np.isfinite(row["ref-minus@0.85"]):
                delta = f"{row['ref-minus@0.85']:+.3f} [{row['ci_low@0.85']:+.3f}, {row['ci_high@0.85']:+.3f}]"
            else:
                delta = "ref" if row["variant"] == reference else "—"
            lines.append(f"| {row['variant']} | " + " | ".join(cells) + f" | {delta} | "
                         f"{row['best_mf']:.3f} ({row['best_p_above']:.2f}, {row['best_diversity']:.2f}) | "
                         f"{row['seconds_per_sample_max']:.3f} | {row['fem_failed']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
