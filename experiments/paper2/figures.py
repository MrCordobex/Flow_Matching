"""Every figure of the sampling-budget sweep, one question per figure.

    uv run python experiments/paper2/figures.py results/budget

Writes results/budget/figures/NN_*.png plus index.html, which lists them with
the question each one answers. Colours follow the evaluator in every figure.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
# Reference categorical slots 1-4, fixed per evaluator across all figures.
EVAL = {
    "NA": ("Noise-aware PB-PUNet", "#2a78d6", "o"),
    "TC": ("Tweedie + surrogate limpio", "#eb6834", "s"),
    "HY": ("Noise-aware híbrido 11M", "#1baf7a", "D"),
    "TS": ("Tweedie + mismos pesos", "#eda100", "^"),
}
KS = [10, 20, 50, 100, 250, 1000]
RNG = np.random.default_rng(0)
FIGURES: list[tuple[str, str, str]] = []  # (file, title, what it answers)


# ---------------------------------------------------------------- data

def load(results: Path) -> dict:
    frontier = {r["run_id"]: r for r in csv.DictReader((results / "frontier.csv").open(encoding="utf-8"))}
    runs = {}
    for entry in json.loads((results / "manifest.json").read_text(encoding="utf-8")):
        mf = np.full(entry["samples"], np.nan)
        metrics = results / "kratos" / entry["slug"] / "metrics.csv"
        for row in csv.DictReader(metrics.open(encoding="utf-8")):
            if row["status"] == "ok":
                mf[int(row["sample_index"])] = float(row["mf_resultants_area_mean"])
        key = (entry["block"], entry["provider"], entry["engineer"], entry["steps"], entry["eta"],
               entry["clip_denoised"], entry["guidance_scale"], entry["guide_every"])
        runs[key] = {"mf": mf, "row": frontier[entry["run_id"]], "trace": entry["trace"],
                     "seconds": entry["seconds"]}
    return runs


def run(runs, name: str, K: int, eta: float = 1.0, clip: bool = True, g: float = 250.0, every: int = 1,
        block: str | None = None):
    provider, engineer, default_block = {
        "NA": ("noise_aware", "pbunet", "b1"), "TC": ("tweedie_clean", "pbunet", "b1"),
        "TS": ("tweedie_self", "pbunet", "b1"), "HY": ("noise_aware", "hybrid", "b4"),
    }[name]
    return runs.get((block or default_block, provider, engineer, K, eta, clip, g, every))


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    boot = RNG.choice(values, (2000, values.size)).mean(1)
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


def paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    return mean_ci(a - b)


def num(r, key: str) -> float:
    return float(r["row"][key])


# ---------------------------------------------------------------- style

def axes(title: str, xlabel: str, ylabel: str, size=(7.2, 4.6), ncols: int = 1, sharey: bool = False):
    figure, grid = plt.subplots(1, ncols, figsize=size, facecolor=SURFACE, sharey=sharey, squeeze=False)
    for axis in grid[0]:
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(INK_2)
        axis.tick_params(colors=INK_2, labelsize=9)
        axis.set_xlabel(xlabel, color=INK, fontsize=10)
    grid[0][0].set_ylabel(ylabel, color=INK, fontsize=10)
    figure.suptitle(title, x=0.02, ha="left", color=INK, fontsize=12, fontweight="bold")
    return figure, (grid[0] if ncols > 1 else grid[0][0])


def k_axis(axis) -> None:
    axis.set_xscale("log")
    axis.set_xticks(KS)
    axis.set_xticklabels([str(k) for k in KS])
    axis.minorticks_off()


def end_label(axis, x, y, text, color) -> None:
    axis.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                  fontsize=8, color=INK_2)
    axis.plot([], [], color=color)


def legend(axis, **kwargs) -> None:
    axis.legend(frameon=False, fontsize=8, labelcolor=INK, **kwargs)


def save(figure, out: Path, name: str, title: str, answers: str) -> None:
    figure.tight_layout()
    figure.savefig(out / name, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    FIGURES.append((name, title, answers))


def line_by_k(axis, runs, names, metric, eta=1.0, ci=True, label_ends=True):
    """metric(run) -> (mean, low, high) or float."""
    for name in names:
        label, color, marker = EVAL[name]
        xs, ys, lo, hi = [], [], [], []
        for K in KS:
            r = run(runs, name, K, eta=eta)
            if r is None:
                continue
            value = metric(r)
            if isinstance(value, tuple):
                ys.append(value[0]); lo.append(value[1]); hi.append(value[2])
            else:
                ys.append(value)
            xs.append(K)
        axis.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2, label=label)
        if ci and lo:
            axis.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        if label_ends and xs:
            end_label(axis, xs[-1], ys[-1], label.split(" ")[0] + (" " + label.split(" ")[-1] if name in ("HY", "TS", "TC") else ""), color)
    k_axis(axis)


# ---------------------------------------------------------------- figures

def fig_quality(runs, out):
    figure, axis = axes("¿Cuánta calidad estructural se pierde al bajar K?",
                        "Pasos de muestreo K (log)", "MF medio (Kratos FEM), IC 95%")
    line_by_k(axis, runs, ["NA", "HY", "TC", "TS"], lambda r: mean_ci(r["mf"]), label_ends=False)
    for g, K in ((0.0, 20), (0.0, 1000)):
        r = run(runs, "NA", K, g=0.0, block="b3")
        axis.scatter([K], [np.nanmean(r["mf"])], marker="x", color=NEUTRAL, s=50, zorder=4)
    axis.scatter([], [], marker="x", color=NEUTRAL, label="Sin guía")
    legend(axis, loc="lower right")
    save(figure, out, "01_mf_vs_K.png", "Calidad frente a presupuesto",
         "MF del FEM en función de K para los cuatro evaluadores (eta=1). Bandas: IC 95% bootstrap.")


def fig_decomposition(runs, out):
    figure, axis = axes("Descomposición: dónde evalúas frente a con qué entrenaste",
                        "Pasos de muestreo K (log)", "Diferencia pareada de MF, IC 95%")
    pairs = [("NA", "TS", "Efecto de dónde evalúas: NA − TS (mismos pesos)", EVAL["NA"][1]),
             ("TS", "TC", "Efecto del entrenamiento: TS − TC (mismo punto)", EVAL["TC"][1]),
             ("NA", "TC", "Total: NA − TC", INK_2)]
    for a, b, label, color in pairs:
        xs, m, lo, hi = [], [], [], []
        for K in KS:
            ra, rb = run(runs, a, K), run(runs, b, K)
            if ra is None or rb is None:
                continue
            value = paired_ci(ra["mf"], rb["mf"])
            xs.append(K); m.append(value[0]); lo.append(value[1]); hi.append(value[2])
        axis.plot(xs, m, marker="o", color=color, linewidth=2, label=label,
                  linestyle="--" if a == "NA" and b == "TC" else "-")
        axis.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
    axis.axhline(0, color=INK, linewidth=1)
    k_axis(axis)
    legend(axis, loc="upper right")
    save(figure, out, "02_decomposition.png", "Descomposición del efecto del proveedor",
         "Por encima de 0 gana el primero. NA−TS aísla el efecto de evaluar en (x_t,t) con los mismos pesos; "
         "TS−TC aísla el efecto de usar un surrogate entrenado solo en limpio.")


def fig_diversity(runs, out):
    figure, axis = axes("¿Cuánta diversidad se pierde al bajar K?", "Pasos de muestreo K (log)",
                        "Distancia media en el latente VAE")
    line_by_k(axis, runs, ["NA", "HY", "TC", "TS"], lambda r: num(r, "latent_mean_distance"),
              ci=False, label_ends=False)
    for K in (20, 1000):
        r = run(runs, "NA", K, g=0.0, block="b3")
        axis.scatter([K], [num(r, "latent_mean_distance")], marker="x", color=NEUTRAL, s=50, zorder=4)
    axis.scatter([], [], marker="x", color=NEUTRAL, label="Sin guía")
    legend(axis, loc="lower right")
    save(figure, out, "03_diversity_vs_K.png", "Diversidad frente a presupuesto",
         "Distancia media entre pares en el latente 2D de la VAE del paper 1. Más alto = más diverso.")


def fig_tail(runs, out, threshold, above, name, title):
    figure, axis = axes(title, "Pasos de muestreo K (log)",
                        f"Fracción con MF {'>' if above else '<'} {threshold}")
    metric = (lambda r: float(np.nanmean(r["mf"] > threshold))) if above else \
             (lambda r: float(np.nanmean(r["mf"] < threshold)))
    line_by_k(axis, runs, ["NA", "HY", "TC", "TS"], metric, ci=False, label_ends=False)
    legend(axis, loc="best")
    save(figure, out, name, title,
         f"Proporción de las 100 láminas con MF {'por encima' if above else 'por debajo'} de {threshold}.")


def fig_percentiles(runs, out):
    figure, grid = axes("Distribución completa del MF por presupuesto", "", "MF (Kratos FEM)",
                        size=(12, 4.4), ncols=3, sharey=True)
    for axis, K in zip(grid, (10, 20, 1000)):
        data, colors, labels = [], [], []
        for name in ("NA", "HY", "TC", "TS"):
            r = run(runs, name, K)
            data.append(r["mf"][~np.isnan(r["mf"])]); colors.append(EVAL[name][1]); labels.append(name)
        parts = axis.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                             medianprops=dict(color=INK, linewidth=1.5),
                             flierprops=dict(marker="o", markersize=3, markerfacecolor=NEUTRAL,
                                             markeredgecolor="none"))
        for patch, color in zip(parts["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor(color)
        axis.set_xticks(range(1, 5)); axis.set_xticklabels(labels)
        axis.set_title(f"K = {K}", color=INK, fontsize=10, loc="left")
    save(figure, out, "06_mf_distribution.png", "Distribución del MF",
         "Cajas: cuartiles; bigotes 1,5·IQR; puntos: láminas atípicas. NA=noise-aware, HY=híbrido, "
         "TC=Tweedie limpio, TS=Tweedie mismos pesos.")


def fig_paired_scatter(runs, out):
    figure, grid = axes("Lámina a lámina: NA frente a Tweedie limpio (mismo ruido inicial)",
                        "MF con Tweedie limpio", "MF con noise-aware", size=(12, 4.4), ncols=3, sharey=True)
    for axis, K in zip(grid, (10, 20, 1000)):
        a, b = run(runs, "NA", K)["mf"], run(runs, "TC", K)["mf"]
        ok = ~np.isnan(a) & ~np.isnan(b)
        axis.scatter(b[ok], a[ok], s=14, color=EVAL["NA"][1], alpha=0.7, edgecolor="none")
        axis.plot([0.3, 1], [0.3, 1], color=INK_2, linewidth=1)
        axis.set_xlim(0.35, 1.0); axis.set_ylim(0.35, 1.0)
        wins = float(np.mean(a[ok] > b[ok]))
        axis.set_title(f"K = {K}: NA gana en {wins:.0%} de las láminas", color=INK, fontsize=10, loc="left")
    save(figure, out, "07_paired_scatter.png", "Comparación pareada lámina a lámina",
         "Cada punto es la misma semilla con dos guías. Por encima de la diagonal gana noise-aware.")


def fig_eta_gain(runs, out):
    figure, axis = axes("¿Cuánto MF aporta la estocasticidad? (eta=1 frente a eta=0)",
                        "Pasos de muestreo K (log)", "MF(eta=1) − MF(eta=0), pareado, IC 95%")
    for name in ("NA", "TC", "TS"):
        label, color, marker = EVAL[name]
        xs, m, lo, hi = [], [], [], []
        for K in KS:
            a, b = run(runs, name, K, eta=1.0), run(runs, name, K, eta=0.0)
            if a is None or b is None:
                continue
            value = paired_ci(a["mf"], b["mf"])
            xs.append(K); m.append(value[0]); lo.append(value[1]); hi.append(value[2])
        axis.plot(xs, m, marker=marker, color=color, linewidth=2, label=label)
        axis.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
    axis.axhline(0, color=INK, linewidth=1)
    k_axis(axis)
    legend(axis, loc="upper right")
    save(figure, out, "08_eta_gain.png", "Ganancia por estocasticidad",
         "Por encima de 0, el sampler estocástico (DDPM) da más MF que el determinista (DDIM).")


def fig_roughness(runs, out):
    figure, axis = axes("El muestreo determinista genera picos de curvatura",
                        "Pasos de muestreo K (log)", "Rugosidad media |∇²z| (cm)")
    for name in ("NA", "TC", "TS"):
        label, color, marker = EVAL[name]
        for eta, style in ((0.0, "--"), (1.0, "-")):
            xs = [K for K in KS if run(runs, name, K, eta=eta)]
            ys = [num(run(runs, name, K, eta=eta), "roughness_mean_cm") for K in xs]
            axis.plot(xs, ys, color=color, marker=marker, linestyle=style, linewidth=2,
                      label=f"{label}, eta={eta:g}")
    real = 2.99
    axis.axhline(real, color=NEUTRAL, linewidth=1.2)
    axis.annotate("láminas reales", (1000, real), xytext=(-4, 4), textcoords="offset points",
                  ha="right", fontsize=8, color=INK_2)
    axis.set_yscale("log")
    ticks = [2, 3, 5, 10, 20, 50]
    axis.set_yticks(ticks); axis.set_yticklabels([str(t) for t in ticks]); axis.minorticks_off()
    k_axis(axis)
    legend(axis, loc="upper right", ncol=2)
    save(figure, out, "09_roughness_vs_K.png", "Rugosidad frente a presupuesto",
         "Línea discontinua: eta=0 (DDIM). Continua: eta=1 (DDPM). La referencia es la rugosidad media del dataset.")


def fig_b2(runs, out):
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    figure, grid = axes("b2: ¿protege el recorte o la estocasticidad? (K=50, noise-aware)",
                        "eta (0 = DDIM, 1 = DDPM)", "", size=(11, 4.2), ncols=2)
    for clip, color, style in ((True, EVAL["NA"][1], "-"), (False, INK_2, "--")):
        rs = [run(runs, "NA", 50, eta=e, clip=clip, block="b2") for e in etas]
        stats = [mean_ci(r["mf"]) for r in rs]
        grid[0].plot(etas, [s[0] for s in stats], marker="o", color=color, linestyle=style, linewidth=2,
                     label=f"recorte {'activado' if clip else 'desactivado'}")
        grid[0].fill_between(etas, [s[1] for s in stats], [s[2] for s in stats], color=color, alpha=0.12)
        grid[1].plot(etas, [num(r, "roughness_mean_cm") for r in rs], marker="o", color=color,
                     linestyle=style, linewidth=2)
    grid[0].set_ylabel("MF medio (FEM), IC 95%", color=INK)
    grid[1].set_ylabel("Rugosidad media (cm)", color=INK)
    legend(grid[0], loc="lower right")
    save(figure, out, "10_b2_eta_clip.png", "Recorte frente a estocasticidad",
         "Las dos líneas coinciden: el recorte de x̂₀ no aporta nada. Todo el efecto viene de eta.")


def fig_b3(runs, out):
    gammas = [0.0, 10.0, 50.0, 250.0, 1000.0]
    figure, grid = axes("b3: escala de guía γ (noise-aware, eta=1)", "γ (× w_max = 8)", "",
                        size=(11, 4.2), ncols=2)
    for K, color, style in ((20, EVAL["NA"][1], "-"), (1000, INK_2, "--")):
        rs = [run(runs, "NA", K, g=g, block="b3") for g in gammas]
        stats = [mean_ci(r["mf"]) for r in rs]
        xs = [max(g, 2.0) for g in gammas]  # 0 drawn at the left edge of the log axis
        grid[0].plot(xs, [s[0] for s in stats], marker="o", color=color, linestyle=style, linewidth=2,
                     label=f"K = {K}")
        grid[0].fill_between(xs, [s[1] for s in stats], [s[2] for s in stats], color=color, alpha=0.12)
        grid[1].plot(xs, [num(r, "latent_mean_distance") for r in rs], marker="o", color=color,
                     linestyle=style, linewidth=2, label=f"K = {K}")
    for axis in grid:
        axis.set_xscale("log")
        axis.set_xticks([2, 10, 50, 250, 1000]); axis.set_xticklabels(["0", "10", "50", "250", "1000"])
        axis.minorticks_off()
    grid[0].set_ylabel("MF medio (FEM), IC 95%", color=INK)
    grid[1].set_ylabel("Diversidad (distancia latente)", color=INK)
    legend(grid[0], loc="lower right")
    save(figure, out, "11_b3_guidance_scale.png", "Escala de guía",
         "La calidad satura desde γ≈50 mientras la diversidad sigue cayendo. Las curvas K=20 y K=1000 casi coinciden: "
         "no hace falta recalibrar γ al bajar K.")


def fig_b4(runs, out):
    figure, axis = axes("b4: el híbrido de 11 M frente a los dos PB-PUNet", "Pasos de muestreo K (log)",
                        "Diferencia pareada de MF, IC 95%")
    for other, color in (("NA", EVAL["NA"][1]), ("TC", EVAL["TC"][1])):
        xs, m, lo, hi = [], [], [], []
        for K in KS:
            value = paired_ci(run(runs, "HY", K)["mf"], run(runs, other, K)["mf"])
            xs.append(K); m.append(value[0]); lo.append(value[1]); hi.append(value[2])
        axis.plot(xs, m, marker="D", color=color, linewidth=2, label=f"híbrido − {EVAL[other][0]}")
        axis.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
    axis.axhline(0, color=INK, linewidth=1)
    k_axis(axis)
    legend(axis, loc="upper right")
    save(figure, out, "12_b4_hybrid.png", "Surrogate híbrido",
         "Por encima de 0 gana el híbrido. Supera al PB-PUNet noise-aware en todos los K y a Tweedie limpio solo a K=10.")


def fig_cost(runs, out):
    figure, axis = axes("Coste de generación", "Pasos de muestreo K (log)",
                        "Segundos por 100 láminas (A100, lote 25, log)")
    line_by_k(axis, runs, ["NA", "HY", "TC", "TS"], lambda r: r["seconds"], ci=False, label_ends=False)
    xs = [20, 1000]
    axis.plot(xs, [run(runs, "NA", K, g=0.0, block="b3")["seconds"] for K in xs], color=NEUTRAL,
              marker="x", linewidth=1.5, label="Sin guía (solo Architect)")
    axis.set_yscale("log")
    legend(axis, loc="upper left")
    save(figure, out, "13_cost_vs_K.png", "Coste frente a presupuesto",
         "Tiempo real medido en la A100. TC y TS cuestan un Architect extra por paso.")


def fig_quality_cost(runs, out):
    figure, axis = axes("Calidad frente a coste (cada punto es un K)", "Segundos por 100 láminas (log)",
                        "MF medio (Kratos FEM)")
    for name in ("NA", "HY", "TC", "TS"):
        label, color, marker = EVAL[name]
        rs = [(K, run(runs, name, K)) for K in KS]
        xs, ys = [r["seconds"] for _, r in rs], [np.nanmean(r["mf"]) for _, r in rs]
        axis.plot(xs, ys, color=color, marker=marker, linewidth=2, label=label)
        for (K, _), x, y in zip(rs, xs, ys):
            if K in (10, 20):
                axis.annotate(f"K{K}", (x, y), xytext=(4, -10), textcoords="offset points",
                              fontsize=7, color=INK_2)
    axis.set_xscale("log")
    legend(axis, loc="lower right")
    save(figure, out, "14_quality_vs_cost.png", "Calidad frente a coste",
         "Arriba a la izquierda es mejor. Cada línea recorre K de izquierda (K=10) a derecha (K=1000). Por debajo de ~20 s solo el híbrido mantiene la calidad.")


def fig_quality_diversity(runs, out):
    figure, grid = axes("Calidad frente a diversidad, un panel por evaluador (la línea recorre K)",
                        "Diversidad (distancia latente)", "MF medio (FEM)", size=(13, 3.8), ncols=4, sharey=True)
    unguided = [run(runs, "NA", K, g=0.0, block="b3") for K in (20, 1000)]
    for axis, name in zip(grid, ("NA", "HY", "TC", "TS")):
        label, color, marker = EVAL[name]
        for other in ("NA", "HY", "TC", "TS"):  # context: the other evaluators in grey
            if other == name:
                continue
            rs = [run(runs, other, K) for K in KS]
            axis.plot([num(r, "latent_mean_distance") for r in rs], [np.nanmean(r["mf"]) for r in rs],
                      color=GRID, linewidth=1.5, zorder=1)
        rs = [(K, run(runs, name, K)) for K in KS]
        xs = [num(r, "latent_mean_distance") for _, r in rs]
        ys = [np.nanmean(r["mf"]) for _, r in rs]
        axis.plot(xs, ys, color=color, marker=marker, linewidth=2, zorder=3)
        for (K, _), x, y in zip(rs, xs, ys):
            if K in (10, 20, 1000):
                axis.annotate(f"K{K}", (x, y), xytext=(4, -11 if K == 1000 else 4), textcoords="offset points",
                              fontsize=7, color=INK_2)
        axis.scatter([num(r, "latent_mean_distance") for r in unguided], [np.nanmean(r["mf"]) for r in unguided],
                     marker="x", color=NEUTRAL, s=40, zorder=3)
        axis.set_title(label, color=INK, fontsize=10, loc="left")
        axis.set_xlim(0.55, 1.7)
    save(figure, out, "15_quality_vs_diversity.png", "Calidad frente a diversidad",
         "Arriba a la derecha es mejor. La línea recorre K = 10, 20, 50, 100, 250, 1000; solo se rotulan 10, 20 y 1000. Gris: los otros evaluadores. Aspas: sin guía (K=20 y K=1000).")


def fig_b5(runs, out):
    everys = [1, 2, 4, 10]
    rs = [run(runs, "NA", 100, every=e, block="b5") for e in everys]
    figure, grid = axes("b5: guiar solo cada n pasos (K=100, noise-aware)", "Guía cada n pasos", "",
                        size=(12, 4.0), ncols=3)
    stats = [mean_ci(r["mf"]) for r in rs]
    grid[0].plot(everys, [s[0] for s in stats], marker="o", color=EVAL["NA"][1], linewidth=2)
    grid[0].fill_between(everys, [s[1] for s in stats], [s[2] for s in stats], color=EVAL["NA"][1], alpha=0.15)
    grid[0].set_ylabel("MF medio (FEM), IC 95%", color=INK)
    grid[1].plot(everys, [num(r, "latent_mean_distance") for r in rs], marker="o", color=EVAL["NA"][1], linewidth=2)
    grid[1].set_ylabel("Diversidad (distancia latente)", color=INK)
    grid[2].bar([str(e) for e in everys], [r["seconds"] for r in rs], color=EVAL["NA"][1], width=0.6)
    grid[2].set_ylabel("Segundos por 100 láminas", color=INK)
    for axis in grid[:2]:
        axis.set_xticks(everys)
    save(figure, out, "16_b5_sparse_guidance.png", "Guía dispersa",
         "Guiar cada 2 pasos mantiene el MF, gana diversidad y cuesta la mitad.")


def sigma_dphi(ts: list[int]) -> np.ndarray:
    def bar(t):
        f = lambda x: math.cos((x / 1000 + 0.008) / 1.008 * math.pi / 2) ** 2
        return f(t) / f(0)
    phi = [math.acos(math.sqrt(bar(t))) for t in ts] + [0.0]
    return np.array([math.sqrt(1 - bar(t)) * (phi[i] - phi[i + 1]) for i, t in enumerate(ts)])


def fig_guidance_timing(runs, out):
    figure, axis = axes("¿Cuándo actúa cada guía? Fracción de la corrección aplicada con ruido bajo (t < 500)",
                        "Pasos de muestreo K (log)", "% de la corrección efectiva en t < 500")
    def late(r):
        ts = [s["t"] for s in r["trace"]]
        eff = np.array([s["grad"] for s in r["trace"]]) * sigma_dphi(ts)
        return 100 * eff[np.array(ts) < 500].sum() / eff.sum()
    line_by_k(axis, runs, ["NA", "HY", "TC", "TS"], late, ci=False, label_ends=False)
    legend(axis, loc="upper right")
    save(figure, out, "17_guidance_timing.png", "Momento de la guía",
         "Los noise-aware corrigen casi todo a ruido alto, que es donde se decide la forma. Tweedie corrige más tarde. "
         "Corrección efectiva = norma × σ_t × Δφ, media del primer lote.")


def fig_guidance_profile(runs, out):
    figure, grid = axes("Perfil de la guía a lo largo de la generación", "Timestep t (ruido alto → limpio)",
                        "Norma de la corrección", size=(12, 4.2), ncols=2)
    for axis, K in zip(grid, (10, 100)):
        for name in ("NA", "HY", "TC", "TS"):
            label, color, marker = EVAL[name]
            trace = run(runs, name, K)["trace"]
            axis.plot([s["t"] for s in trace], [s["grad"] for s in trace], color=color, linewidth=2,
                      marker=marker if K == 10 else None, markersize=5, label=label)
        axis.invert_xaxis()
        axis.set_title(f"K = {K}", color=INK, fontsize=10, loc="left")
    legend(grid[0], loc="upper left")
    save(figure, out, "18_guidance_profile.png", "Perfil temporal de la guía",
         "Norma de la corrección aplicada en cada paso (media del primer lote de 25 láminas).")


def fig_predicted_mf(runs, out):
    figure, grid = axes("MF que cree cada surrogate durante la generación", "Timestep t (ruido alto → limpio)",
                        "MF predicho por su propio surrogate", size=(12, 4.2), ncols=2, sharey=True)
    for axis, K in zip(grid, (10, 100)):
        for name in ("NA", "HY", "TC", "TS"):
            label, color, marker = EVAL[name]
            trace = run(runs, name, K)["trace"]
            axis.plot([s["t"] for s in trace], [s["mf"] for s in trace], color=color, linewidth=2, label=label)
            fem = np.nanmean(run(runs, name, K)["mf"])
            axis.axhline(fem, color=color, linewidth=1, linestyle=":")
        axis.invert_xaxis()
        axis.set_title(f"K = {K}  (punteado: MF real del FEM)", color=INK, fontsize=10, loc="left")
    legend(grid[0], loc="lower right")
    save(figure, out, "19_predicted_mf_trace.png", "Lo que cree el surrogate",
         "Línea continua: MF que el surrogate de guía predice en cada paso. Punteada: MF real de Kratos al final. "
         "La distancia entre ambas al final es el sesgo de autoevaluación.")


def fig_bias(runs, out):
    figure, axis = axes("Sesgo de autoevaluación: MF predicho al final − MF real del FEM",
                        "Pasos de muestreo K (log)", "Sesgo (predicho − FEM)")
    line_by_k(axis, runs, ["NA", "HY", "TC", "TS"],
              lambda r: r["trace"][-1]["mf"] - float(np.nanmean(r["mf"])), ci=False, label_ends=False)
    axis.axhline(0, color=INK, linewidth=1)
    legend(axis, loc="upper right")
    save(figure, out, "20_self_assessment_bias.png", "Sesgo del surrogate",
         "Cuánto sobreestima cada surrogate el MF de las láminas que él mismo ha guiado. Aproximado: último paso del "
         "trace, primer lote.")


def fig_convergence(runs, out):
    figure, axis = axes("Fallos de convergencia del FEM", "Pasos de muestreo K (log)",
                        "Láminas que Kratos no resuelve (de 100)")
    for name in ("NA", "TC", "TS"):
        label, color, marker = EVAL[name]
        for eta, style in ((0.0, "--"), (1.0, "-")):
            xs = [K for K in KS if run(runs, name, K, eta=eta)]
            axis.plot(xs, [int(np.isnan(run(runs, name, K, eta=eta)["mf"]).sum()) for K in xs], color=color,
                      marker=marker, linestyle=style, linewidth=2, label=f"{label}, eta={eta:g}")
    k_axis(axis)
    legend(axis, loc="upper right", ncol=2)
    save(figure, out, "21_fem_failures.png", "Convergencia del FEM",
         "Los fallos se concentran a K=10 con eta=0. Las medias de MF excluyen estas láminas.")


def fig_gamma_diversity_tradeoff(runs, out):
    figure, axis = axes("Compromiso calidad-diversidad al mover γ", "Diversidad (distancia latente)",
                        "MF medio (FEM)")
    for K, color, style in ((20, EVAL["NA"][1], "-"), (1000, INK_2, "--")):
        rs = [(g, run(runs, "NA", K, g=g, block="b3")) for g in (0.0, 10.0, 50.0, 250.0, 1000.0)]
        xs, ys = [num(r, "latent_mean_distance") for _, r in rs], [np.nanmean(r["mf"]) for _, r in rs]
        axis.plot(xs, ys, color=color, linestyle=style, marker="o", linewidth=2, label=f"noise-aware, K = {K}")
        for (g, _), x, y in zip(rs, xs, ys):
            if K == 1000 or g in (0.0, 10.0, 1000.0):
                axis.annotate(f"γ{g:g}", (x, y), xytext=(4, 7) if K == 1000 else (-6, -14),
                              textcoords="offset points", fontsize=7, color=INK_2)
    for name, offsets in (("TC", {20: (6, 4), 1000: (6, 4)}), ("HY", {20: (-4, 10), 1000: (8, -12)})):
        for K in (20, 1000):
            r = run(runs, name, K)
            x, y = num(r, "latent_mean_distance"), np.nanmean(r["mf"])
            axis.scatter([x], [y], color=EVAL[name][1], marker=EVAL[name][2], s=55, zorder=4,
                         label=f"{EVAL[name][0]} (γ250)" if K == 20 else None)
            axis.annotate(f"K{K}", (x, y), xytext=offsets[K], textcoords="offset points", fontsize=7, color=INK_2)
    legend(axis, loc="lower left")
    save(figure, out, "22_gamma_tradeoff.png", "γ como palanca calidad-diversidad",
         "Recorrido de γ para noise-aware, con TC e híbrido (γ=250) como referencia. Permite ver a qué γ tendría que "
         "ir NA para igualar a TC.")


# ---------------------------------------------------------------- index

def write_index(out: Path) -> None:
    cards = "\n".join(
        f'<figure><h2>{i + 1:02d}. {title}</h2><img src="{name}" alt="{title}"><figcaption>{text}</figcaption></figure>'
        for i, (name, title, text) in enumerate(FIGURES))
    (out / "index.html").write_text(f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Figuras del barrido</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px}}
figure{{margin:0 0 40px;border-top:1px solid var(--line);padding-top:16px}}
h1{{font-size:22px}} h2{{font-size:16px;margin:0 0 8px}}
img{{width:100%;height:auto;display:block}} figcaption{{color:var(--ink2);font-size:14px;margin-top:6px}}
</style></head><body><main><h1>Barrido de presupuesto: 66 runs × 100 láminas verificadas con Kratos</h1>
{cards}</main></body></html>""", encoding="utf-8")


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/budget")
    out = results / "figures"
    out.mkdir(exist_ok=True)
    runs = load(results)
    fig_quality(runs, out)
    fig_decomposition(runs, out)
    fig_diversity(runs, out)
    fig_tail(runs, out, 0.7, False, "04_tail_below_070.png", "Láminas malas: fracción con MF < 0.7")
    fig_tail(runs, out, 0.9, True, "05_share_above_090.png", "Láminas buenas: fracción con MF > 0.9")
    fig_percentiles(runs, out)
    fig_paired_scatter(runs, out)
    fig_eta_gain(runs, out)
    fig_roughness(runs, out)
    fig_b2(runs, out)
    fig_b3(runs, out)
    fig_b4(runs, out)
    fig_cost(runs, out)
    fig_quality_cost(runs, out)
    fig_quality_diversity(runs, out)
    fig_b5(runs, out)
    fig_guidance_timing(runs, out)
    fig_guidance_profile(runs, out)
    fig_predicted_mf(runs, out)
    fig_bias(runs, out)
    fig_convergence(runs, out)
    fig_gamma_diversity_tradeoff(runs, out)
    write_index(out)
    print(f"wrote {len(FIGURES)} figures and {out / 'index.html'}")


if __name__ == "__main__":
    main()
