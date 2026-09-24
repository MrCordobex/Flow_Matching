"""The preregistered tests of H1-H3, exactly as PREREGISTRO.md section 6 states them.

    uv run python experiments/paper2/hypotheses.py results/budget [--failures-as-zero]

Needs guidance_samples.csv from guidance_metrics.py. Prints each test with its
intervals and a verdict; writes hypotheses.json. Nothing here is tuned: the
thresholds are the preregistered ones.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

GAMMAS = (10.0, 25.0, 50.0, 100.0)
STEPS = (10, 20, 100)
RESAMPLES = 10_000
RNG = np.random.default_rng(20260924)


def load(results: Path, failures_as_zero: bool) -> dict:
    table: dict[tuple, dict[str, np.ndarray]] = {}
    rows = list(csv.DictReader((results / "guidance_samples.csv").open(encoding="utf-8")))
    for row in rows:
        if row["block"] != "b6":
            continue
        key = (row["evaluator"], int(row["steps"]), float(row["gamma"]))
        cell = table.setdefault(key, {"fem_mf": [], "push": [], "straightness": [], "sample": []})
        mf = float(row["fem_mf"]) if row["fem_mf"] not in ("", "nan") else np.nan
        if failures_as_zero and np.isnan(mf):
            mf = 0.0
        cell["fem_mf"].append(mf)
        cell["push"].append(float(row["push"]))
        cell["straightness"].append(float(row["straightness"]))
        cell["sample"].append(int(row["sample"]))
    for cell in table.values():
        order = np.argsort(cell["sample"])
        for key in list(cell):
            cell[key] = np.asarray(cell[key], dtype=float)[order]
    return table


def paired_ci(diff: np.ndarray) -> tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    boot = RNG.choice(diff, (RESAMPLES, diff.size)).mean(1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def h1(table: dict) -> dict:
    """Budget loss L(K) = mf(K) - mf(100), NA against TC, paired by sample."""
    out, gammas_passing = {}, 0
    for gamma in GAMMAS:
        passes = []
        for steps in (10, 20):
            needed = [("NA", steps, gamma), ("NA", 100, gamma), ("TC", steps, gamma), ("TC", 100, gamma)]
            if any(k not in table for k in needed):
                out[f"g{gamma:g}_K{steps}"] = "missing"
                passes.append(False)
                continue
            loss_na = table[needed[0]]["fem_mf"] - table[needed[1]]["fem_mf"]
            loss_tc = table[needed[2]]["fem_mf"] - table[needed[3]]["fem_mf"]
            mean, low, high = paired_ci(loss_na - loss_tc)
            out[f"g{gamma:g}_K{steps}"] = {"diff": mean, "ci": [low, high], "pass": low > 0}
            passes.append(low > 0)
        gammas_passing += all(passes)
    out["gammas_passing"] = gammas_passing
    incomplete = any(v == "missing" for v in out.values())
    out["verdict"] = "incomplete" if incomplete else "supported" if gammas_passing >= 3 else "not supported"
    return out


def h2(table: dict) -> dict:
    """Straightness NA - TC in every (gamma, K) pair."""
    out, favour_na, favour_tc, tested = {}, 0, 0, 0
    for gamma in GAMMAS:
        for steps in STEPS:
            a, b = table.get(("NA", steps, gamma)), table.get(("TC", steps, gamma))
            if a is None or b is None:
                out[f"g{gamma:g}_K{steps}"] = "missing"
                continue
            tested += 1
            mean, low, high = paired_ci(a["straightness"] - b["straightness"])
            favour_na += low > 0
            favour_tc += high < 0
            out[f"g{gamma:g}_K{steps}"] = {"diff": mean, "ci": [low, high]}
    out |= {"pairs": tested, "favour_na": favour_na, "favour_tc": favour_tc}
    out["verdict"] = ("incomplete" if tested < 12 else
                      "supported" if favour_tc == 0 and favour_na >= 10 else "not supported")
    return out


def _curve(table: dict, name: str, steps: int, index: np.ndarray | None = None):
    pushes, mfs = [], []
    for gamma in GAMMAS:
        cell = table[(name, steps, gamma)]
        take = slice(None) if index is None else index
        pushes.append(np.nanmean(cell["push"][take]))
        mfs.append(np.nanmean(cell["fem_mf"][take]))
    order = np.argsort(pushes)
    return np.asarray(pushes)[order], np.asarray(mfs)[order]


def h3(table: dict) -> dict:
    """MF against total push, NA against TC, at five points of the overlapping push range."""
    out, per_k = {}, {}
    for steps in STEPS:
        if any((n, steps, g) not in table for n in ("NA", "TC") for g in GAMMAS):
            per_k[steps] = "missing"
            continue
        push_na, _ = _curve(table, "NA", steps)
        push_tc, _ = _curve(table, "TC", steps)
        low, high = max(push_na.min(), push_tc.min()), min(push_na.max(), push_tc.max())
        if high <= low or (high - low) < 0.3 * max(np.ptp(push_na), np.ptp(push_tc)):
            per_k[steps] = {"verdict": "not testable", "overlap": [float(low), float(high)]}
            continue
        grid = np.linspace(low, high, 5)
        size = len(table[("NA", steps, GAMMAS[0])]["fem_mf"])
        boot = np.empty((RESAMPLES, grid.size))
        for b in range(boot.shape[0]):  # resample sample indices jointly: pairing survives
            index = RNG.integers(0, size, size)
            pn, mn = _curve(table, "NA", steps, index)
            pt, mt = _curve(table, "TC", steps, index)
            boot[b] = np.interp(grid, pn, mn) - np.interp(grid, pt, mt)
        pn, mn = _curve(table, "NA", steps)
        pt, mt = _curve(table, "TC", steps)
        point = np.interp(grid, pn, mn) - np.interp(grid, pt, mt)
        lower = np.percentile(boot, 2.5, axis=0)
        verdict = "superior" if (lower > 0).all() else "non-inferior" if (lower > -0.01).all() else "inferior"
        per_k[steps] = {"push_grid": grid.tolist(), "diff": point.tolist(), "ci_low": lower.tolist(),
                        "ci_high": np.percentile(boot, 97.5, axis=0).tolist(), "verdict": verdict}
    out["per_K"] = per_k
    ok = all(isinstance(per_k.get(k), dict) and per_k[k].get("verdict") in ("superior", "non-inferior")
             for k in (100, 20))
    if any(per_k.get(k) == "missing" for k in (100, 20)):
        out["verdict"] = "incomplete"
    else:
        out["verdict"] = "supported" if ok else "not supported"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--failures-as-zero", action="store_true",
                        help="the preregistered sensitivity analysis for H1 and H3")
    args = parser.parse_args()
    table = load(args.results, args.failures_as_zero)
    report = {"failures_as_zero": args.failures_as_zero, "H1": h1(table), "H2": h2(table), "H3": h3(table)}
    suffix = "_failures_zero" if args.failures_as_zero else ""
    (args.results / f"hypotheses{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("H1", "H2", "H3"):
        print(f"{name}: {report[name]['verdict']}")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
