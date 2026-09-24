"""Calibration of every guidance evaluator against noise level (results/calibration).

    uv run python experiments/paper2/plot_calibration.py results/calibration results/budget/figures
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "noise_aware": ("Noise-aware PB-PUNet, en (x_t, t)", "#2a78d6", "o", "-"),
    "tweedie_self": ("Mismos pesos, en (x̂₀, 0)", "#eda100", "^", "-"),
    "tweedie_clean": ("Surrogate limpio, en (x̂₀, 0)", "#eb6834", "s", "-"),
    "hybrid": ("Híbrido, en (x_t, t)", "#1baf7a", "D", "-"),
    "hybrid_tweedie": ("Híbrido, en (x̂₀, 0)", "#1baf7a", "D", "--"),
}


def main() -> None:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader((source / "calibration_summary.csv").open(encoding="utf-8")))
    panels = (("field_mse", "Error de los campos (MSE normalizado, log)", True),
              ("mf_bias_high", "Sesgo del MF predicho (MF real ≥ 0.85)", False))
    figure, grid = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    for axis, (key, ylabel, log) in zip(grid, panels):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=INK_2, labelsize=9)
        for name, (label, color, marker, line) in STYLE.items():
            chosen = [r for r in rows if r["evaluator"] == name]
            axis.plot([int(r["t"]) for r in chosen], [float(r[key]) for r in chosen], color=color,
                      marker=marker, markersize=5, linestyle=line, linewidth=2, label=label)
        if log:
            axis.set_yscale("log")
            ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            axis.set_yticks(ticks); axis.set_yticklabels([f"{v:g}" for v in ticks]); axis.minorticks_off()
        else:
            axis.axhline(0, color=INK, linewidth=1)
        axis.set_xlabel("Timestep t (0 = limpio, 999 = ruido puro)", color=INK, fontsize=10)
        axis.set_ylabel(ylabel, color=INK, fontsize=10)
    grid[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    figure.suptitle("Calibración sin muestrear: 120 láminas de validación, mismos estados ruidosos para todos",
                    x=0.02, ha="left", color=INK, fontsize=12, fontweight="bold")
    figure.tight_layout()
    figure.savefig(out / "23_calibration_vs_t.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out / "23_calibration_vs_t.png")


if __name__ == "__main__":
    main()
