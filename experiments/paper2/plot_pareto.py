"""Pareto fronts of the sampling-budget sweep, from frontier.csv.

(a) FEM-verified MF against latent diversity, (b) MF against wall-clock cost.
Only stochastic, clipped runs (eta=1, clip=True) are drawn: eta=0 is dominated
everywhere and would only crowd the fronts.

    uv run python experiments/paper2/plot_pareto.py results/budget
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical slots 1-3 (validated all-pairs, light mode); the fourth
# evaluator is drawn as neutral context instead of taking a fourth hue.
SURFACE, INK, INK_2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
SERIES = {
    "noise_aware": ("Noise-aware (PB-PUNet)", "#2a78d6", "o"),
    "tweedie_clean": ("Tweedie, clean surrogate", "#eb6834", "s"),
    "hybrid": ("Noise-aware (hybrid, 11 M)", "#1baf7a", "D"),
    "tweedie_self": ("Tweedie, same weights", CONTEXT, "^"),
}


def load(results: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    rows = list(csv.DictReader((results / "frontier.csv").open(encoding="utf-8")))
    series: dict[str, list[dict]] = {key: [] for key in SERIES}
    unguided, seen = [], set()
    for row in rows:
        if float(row["eta"]) != 1.0 or row["clip_denoised"] != "True":
            continue
        config = (row["provider"], row["engineer"], row["steps"], row["guidance_scale"], row["guide_every"])
        if config in seen:  # b2/b3/b5 repeat some b1 configurations
            continue
        seen.add(config)
        point = {
            "mf": float(row["fem_mf_mean"]), "div": float(row["latent_mean_distance"]),
            "cost": float(row["seconds"]), "K": int(row["steps"]),
            "g": float(row["guidance_scale"]), "every": int(row["guide_every"]),
        }
        if point["g"] == 0.0:
            unguided.append(point)
        elif row["engineer"] == "hybrid":
            series["hybrid"].append(point)
        else:
            series[row["provider"]].append(point)
    return series, unguided


def front(points: list[dict], x: str, maximise_x: bool) -> list[dict]:
    """Non-dominated points when MF is maximised and x is maximised or minimised."""
    better = (lambda a, b: a >= b) if maximise_x else (lambda a, b: a <= b)
    kept = [p for p in points
            if not any(q is not p and q["mf"] >= p["mf"] and better(q[x], p[x])
                       and (q["mf"] > p["mf"] or q[x] != p[x]) for q in points)]
    return sorted(kept, key=lambda p: p[x])


def label(point: dict) -> str:
    text = f"K{point['K']}"
    if point["g"] != 250.0:
        text += f" γ{point['g']:g}"
    if point["every"] != 1:
        text += f" /{point['every']}"
    return text


def style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_2)
    axis.tick_params(colors=INK_2, labelsize=9)


class Labeller:
    """Places direct labels, skipping any that would land on one already drawn."""

    def __init__(self, axis: plt.Axes, gap_x: float = 70.0, gap_y: float = 26.0) -> None:
        self.axis, self.gap_x, self.gap_y, self.placed = axis, gap_x, gap_y, []

    def __call__(self, text: str, x: float, y: float, dx: float, dy: float) -> None:
        px, py = self.axis.transData.transform((x, y))
        scale = self.axis.figure.dpi / 72.0  # offsets are in points
        px, py = px + dx * scale, py + dy * scale
        if any(abs(px - qx) < self.gap_x and abs(py - qy) < self.gap_y for qx, qy in self.placed):
            return
        self.placed.append((px, py))
        self.axis.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                           fontsize=7, color=INK_2)


def panel(axis: plt.Axes, series: dict, unguided: list[dict], x: str, maximise_x: bool) -> None:
    style(axis)
    if x == "cost":
        axis.set_xscale("log")
    axis.set_xlim(*( (0.55, 1.72) if x == "div" else (2.2, 1500) ))
    axis.set_ylim(0.53, 0.97)
    place = Labeller(axis)
    for key, (name, color, marker) in SERIES.items():
        points = series[key]
        if not points:
            continue
        context = key == "tweedie_self"
        axis.scatter([p[x] for p in points], [p["mf"] for p in points], s=34, marker=marker,
                     facecolor="none" if context else color, edgecolor=color, linewidth=1.4,
                     zorder=3, label=name)
        edge = front(points, x, maximise_x)
        axis.plot([p[x] for p in edge], [p["mf"] for p in edge], color=color, linewidth=2,
                  linestyle=":" if context else "-", zorder=2)
        for p in edge:
            place(label(p), p[x], p["mf"], 5, -9)
    axis.scatter([p[x] for p in unguided], [p["mf"] for p in unguided], s=40, marker="x",
                 color=INK, linewidth=1.4, zorder=3, label="Unguided")
    for p in unguided:
        place(f"K{p['K']}", p[x], p["mf"], 5, 3)
    axis.set_ylabel("Membrane factor, Kratos FEM", color=INK, fontsize=10)


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/budget")
    series, unguided = load(results)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE)
    figure.set_dpi(200)  # labels are collision-checked in display pixels at the saved resolution

    panel(left, series, unguided, "div", maximise_x=True)
    left.set_xlabel("Diversity: mean pairwise latent distance", color=INK, fontsize=10)
    left.set_title("(a) Quality against diversity", loc="left", color=INK, fontsize=11)

    panel(right, series, unguided, "cost", maximise_x=False)
    right.set_xlabel("Wall-clock seconds per 100 shells (A100, batch 25, log)", color=INK, fontsize=10)
    right.set_title("(b) Quality against cost", loc="left", color=INK, fontsize=11)

    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9,
                  labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    figure.text(0.5, -0.06, "Points: eta=1, clip on, 100 FEM-verified shells each. Lines join each "
                "evaluator's non-dominated runs. Labels: steps K, guidance scale γ when not 250, "
                "/n = guidance every n steps.", ha="center", fontsize=8, color=INK_2)
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    out = results / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"pareto.{suffix}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out / 'pareto.png'} and .pdf")


if __name__ == "__main__":
    main()
