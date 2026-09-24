"""Offline calibration of every guidance evaluator as a function of noise level.

No sampling and no FEM: held-out shells are pushed through the forward process
at fixed timesteps, and each evaluator's Membrane-Factor prediction is compared
with the MF of the true FEM fields of the clean shell. Every evaluator sees the
same noisy states (paired), so differences are differences between evaluators.

    noise_aware     PB-PUNet on (x_t, t)
    tweedie_self    PB-PUNet on (x0_hat, 0)
    tweedie_clean   clean-trained PB-PUNet on (x0_hat, 0)
    hybrid          spectral surrogate on (x_t, t)
    hybrid_tweedie  spectral surrogate on (x0_hat, 0)

x0_hat is the Architect's clipped Tweedie estimate, exactly as the guidance
providers compute it. The target is R(x0), so the MSE-optimal prediction is
E[R(x0) | x_t]: part of the error at high noise is irreducible and shared by all.

    python run_calibration.py --models-dir models --data ../solid_npz.zip --output results/calibration
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from tfm_shells.constants import PHYSICS_KEYS
from tfm_shells.utils.physics import compute_membrane_factor_map_from_real_physics

from paper2.discrete_vp import split_velocity
from paper2.models import architect_to_surrogate, load_architect, load_surrogate

# Split of every published checkpoint (their dataset_summary: 1920 train / 480 val).
SPLIT_SEED, VAL_RATIO = 42, 0.20
VAL_MF_MEAN = 0.6589663158981128
DEFAULT_TIMESTEPS = (0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)
HIGH_MF = 0.85  # the regime guidance drives samples into

# evaluator -> (surrogate key, input, timestep fed to the surrogate)
EVALUATORS = {
    "noise_aware": ("pbunet", "state", "t"),
    "tweedie_self": ("pbunet", "estimate", "zero"),
    "tweedie_clean": ("clean", "estimate", "zero"),
    "hybrid": ("hybrid", "state", "t"),
    "hybrid_tweedie": ("hybrid", "estimate", "zero"),
}
CHECKPOINTS = {"pbunet": "engineer_solid.pt", "clean": "clean_solid.pt", "hybrid": "engineer_hybrid.pt"}
BRANCHES = {"u": slice(0, 1), "m": slice(1, 7), "f": slice(7, 13)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="solid_npz.zip or a directory of shell_*.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-shells", type=int, default=120,
                        help="half are the highest-MF validation shells, half drawn at random from the rest")
    parser.add_argument("--timesteps", type=int, nargs="*", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--evaluators", nargs="*", default=list(EVALUATORS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class ShellSource:
    """Reads shells from the zip as shipped, or from an extracted directory."""

    def __init__(self, path: Path) -> None:
        self.zip = zipfile.ZipFile(path) if path.suffix == ".zip" else None
        self.root = path
        names = self.zip.namelist() if self.zip else [p.name for p in path.glob("*.npz")]
        self.names = sorted(Path(n).name for n in names
                            if n.endswith(".npz") and "shell_hole" not in n)
        self._members = {Path(n).name: n for n in (self.zip.namelist() if self.zip else [])}

    def load(self, name: str) -> dict[str, np.ndarray]:
        if self.zip:
            raw = io.BytesIO(self.zip.read(self._members[name]))
        else:
            raw = self.root / name
        with np.load(raw) as data:
            return {key: np.asarray(data[key]) for key in data.files}


def validation_shells(source: ShellSource, count: int) -> list[tuple[str, dict]]:
    # Same call as tfm_shells.data.index.split_records on the sorted solid list,
    # so this reproduces the held-out set no checkpoint was trained on.
    _, val = train_test_split(source.names, test_size=VAL_RATIO, random_state=SPLIT_SEED, shuffle=True)
    shells = [(name, source.load(name)) for name in val]
    mf_means = np.array([float(data["mf"].mean()) for _, data in shells])
    print(f"validation split: {len(val)} shells, dataset mf mean {mf_means.mean():.10f} "
          f"(checkpoints report {VAL_MF_MEAN:.10f})", flush=True)
    if abs(mf_means.mean() - VAL_MF_MEAN) > 1e-6:
        raise RuntimeError("validation split does not match the checkpoints' split")

    order = np.argsort(-mf_means)
    high = list(order[: count // 2])
    rest = np.random.default_rng(0).permutation(order[count // 2:])[: count - len(high)]
    return [shells[i] for i in sorted(high + list(rest))]


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batched(total: int, size: int):
    for begin in range(0, total, size):
        yield slice(begin, min(begin + size, total))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    shells = validation_shells(ShellSource(args.data), args.n_shells)
    architect, schedule, architect_stats = load_architect(args.models_dir / "architect_solid.pt", device)
    low, high = float(architect_stats["z_min"]), float(architect_stats["z_max"])

    clean = torch.from_numpy(np.stack([2.0 * (d["z"].astype(np.float32) - low) / (high - low) - 1.0
                                       for _, d in shells])).reshape(len(shells), 1, 64, 64)
    physics = torch.from_numpy(np.stack([np.concatenate([d[k].astype(np.float32) for k in PHYSICS_KEYS], 0)
                                         for _, d in shells]))
    fz = torch.from_numpy(np.stack([d["fz"].astype(np.float32) for _, d in shells])).reshape(len(shells), 1, 64, 64)
    mf_true = compute_membrane_factor_map_from_real_physics(physics).mean(dim=(1, 2, 3)).numpy()
    print(f"{len(shells)} shells, true MF in [{mf_true.min():.3f}, {mf_true.max():.3f}], "
          f"{(mf_true >= HIGH_MF).sum()} with MF >= {HIGH_MF}", flush=True)

    # Stage A: one noisy state per (shell, t), shared by every evaluator.
    grid = [(s, t) for t in args.timesteps for s in range(len(shells))]
    states = torch.empty(len(grid), 1, 64, 64)
    estimates = torch.empty_like(states)
    started = time.perf_counter()
    for chunk in batched(len(grid), args.batch_size):
        index = grid[chunk]
        shell_ids = [s for s, _ in index]
        noise = torch.stack([
            torch.randn(1, 64, 64, generator=torch.Generator().manual_seed(1000 * s + t)) for s, t in index])
        alpha = torch.tensor([float(schedule.alpha_sigma(t)[0]) for _, t in index]).view(-1, 1, 1, 1)
        sigma = torch.tensor([float(schedule.alpha_sigma(t)[1]) for _, t in index]).view(-1, 1, 1, 1)
        state = alpha * clean[shell_ids] + sigma * noise
        timesteps = torch.tensor([float(t) for _, t in index])
        velocity = architect(state.to(device), timesteps.to(device)).sample.float().cpu()
        estimate, _ = split_velocity(state, velocity, alpha, sigma, clip_denoised=True)
        states[chunk], estimates[chunk] = state, estimate
    print(f"architect: {len(grid)} states in {time.perf_counter() - started:.0f}s", flush=True)
    del architect
    gc.collect()

    estimate_error = ((estimates - clean[[s for s, _ in grid]]).flatten(1).square().mean(1).sqrt()
                      * 0.5 * (high - low)).numpy()
    rows: list[dict] = []
    for key in dict.fromkeys(EVALUATORS[name][0] for name in args.evaluators):
        path = args.models_dir / CHECKPOINTS[key]
        if not path.exists():
            print(f"skip {key}: {path} missing", flush=True)
            continue
        surrogate = load_surrogate(key, path, device)
        load_field = 2.0 * (fz - float(surrogate.stats["fz_min"])) / (
            float(surrogate.stats["fz_max"]) - float(surrogate.stats["fz_min"]) + 1e-8) - 1.0
        target = ((physics - surrogate.physics_mean.cpu().view(1, -1, 1, 1))
                  / surrogate.physics_std.cpu().view(1, -1, 1, 1))
        for name in (n for n in args.evaluators if EVALUATORS[n][0] == key):
            _, source, clock = EVALUATORS[name]
            inputs = states if source == "state" else estimates
            started = time.perf_counter()
            for chunk in batched(len(grid), args.batch_size):
                index = grid[chunk]
                shell_ids = [s for s, _ in index]
                heights = architect_to_surrogate(inputs[chunk], architect_stats, surrogate.stats)
                timesteps = torch.tensor([float(t) if clock == "t" else 0.0 for _, t in index])
                sample = torch.cat([heights, load_field[shell_ids]], dim=1).to(device)
                prediction = surrogate.model(sample, timesteps.to(device)).sample.float()
                real = prediction * surrogate.physics_std + surrogate.physics_mean
                mf_pred = compute_membrane_factor_map_from_real_physics(real).mean(dim=(1, 2, 3)).cpu().numpy()
                error = (prediction.cpu() - target[shell_ids]).square()
                for row, (s, t) in enumerate(index):
                    rows.append({
                        "evaluator": name, "shell": shells[s][0], "t": t,
                        "mf_true": float(mf_true[s]), "mf_pred": float(mf_pred[row]),
                        "estimate_rmse_m": float(estimate_error[chunk.start + row]),
                        **{f"mse_{b}": float(error[row, part].mean()) for b, part in BRANCHES.items()},
                    })
            print(f"{name}: {len(grid)} evaluations in {time.perf_counter() - started:.0f}s", flush=True)
        del surrogate
        gc.collect()

    with (args.output / "calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarise(rows, args.output / "calibration_summary.csv")


def summarise(rows: list[dict], target: Path) -> None:
    summary = []
    for name in dict.fromkeys(r["evaluator"] for r in rows):
        for t in sorted({r["t"] for r in rows}):
            chosen = [r for r in rows if r["evaluator"] == name and r["t"] == t]
            error = np.array([r["mf_pred"] - r["mf_true"] for r in chosen])
            high = np.array([r["mf_pred"] - r["mf_true"] for r in chosen if r["mf_true"] >= HIGH_MF])
            summary.append({
                "evaluator": name, "t": t, "n": len(chosen),
                "mf_bias": error.mean(), "mf_mae": np.abs(error).mean(),
                "mf_bias_high": high.mean() if high.size else float("nan"),
                "mf_mae_high": np.abs(high).mean() if high.size else float("nan"),
                "field_mse": np.mean([(r["mse_u"] + r["mse_m"] + r["mse_f"]) / 3 for r in chosen]),
                "mse_f": np.mean([r["mse_f"] for r in chosen]),
                "estimate_rmse_m": np.mean([r["estimate_rmse_m"] for r in chosen]),
            })
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'evaluator':<16}{'t':>5}{'bias':>8}{'MAE':>7}{'bias_hi':>9}{'MAE_hi':>8}{'fieldMSE':>10}{'mse_f':>8}{'x0hat_m':>9}")
    for row in summary:
        print(f"{row['evaluator']:<16}{row['t']:>5}{row['mf_bias']:>+8.3f}{row['mf_mae']:>7.3f}"
              f"{row['mf_bias_high']:>+9.3f}{row['mf_mae_high']:>8.3f}{row['field_mse']:>10.4f}"
              f"{row['mse_f']:>8.4f}{row['estimate_rmse_m']:>9.3f}")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
