from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any

from tfm_shells.utils.matplotlib_backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import torch

from tfm_shells.config import config_name, project_root, save_config
from tfm_shells.utils.io import ensure_dir, save_history_csv, save_json, timestamp


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_run_directories(config: dict[str, Any], role: str) -> dict[str, Path]:
    root = project_root(config)
    ensure_dir(root / "artifacts")
    ensure_dir(root / "models")
    ensure_dir(root / "mlruns")

    run_stamp = timestamp()
    run_root = ensure_dir(root / "artifacts" / role / f"{run_stamp}_{config_name(config)}")
    model_root = ensure_dir(root / "models" / role / f"{run_stamp}_{config_name(config)}")
    latest_root = root / "models" / role / "latest"

    return {
        "project_root": root,
        "run_root": run_root,
        "model_root": model_root,
        "latest_root": latest_root,
    }


def finalize_latest_symlink(latest_root: Path, model_root: Path) -> None:
    latest_root.parent.mkdir(parents=True, exist_ok=True)
    if latest_root.exists() or latest_root.is_symlink():
        if latest_root.is_dir() and not latest_root.is_symlink():
            shutil.rmtree(latest_root)
        else:
            latest_root.unlink()
    shutil.copytree(model_root, latest_root)


def save_run_metadata(
    config: dict[str, Any],
    directories: dict[str, Path],
    stats: dict[str, Any],
    splits: dict[str, Any],
) -> None:
    save_config(config, directories["run_root"] / "config.yaml")
    save_config(config, directories["model_root"] / "config.yaml")
    save_json(stats, directories["run_root"] / "stats.json")
    save_json(stats, directories["model_root"] / "stats.json")
    save_json(splits, directories["run_root"] / "splits.json")
    save_json(splits, directories["model_root"] / "splits.json")


def save_history(history_rows: list[dict[str, Any]], directories: dict[str, Path]) -> None:
    save_history_csv(history_rows, directories["run_root"] / "history.csv")
    save_history_csv(history_rows, directories["model_root"] / "history.csv")


def plot_training_curves(
    history_rows: list[dict[str, Any]],
    metrics: list[tuple[str, str | None]],
    title: str,
    output_path: Path,
) -> None:
    if not history_rows:
        return

    epochs = [row["epoch"] for row in history_rows]
    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for axis, (train_key, val_key) in zip(axes, metrics):
        axis.plot(epochs, [row[train_key] for row in history_rows], label=train_key)
        if val_key is not None:
            axis.plot(epochs, [row[val_key] for row in history_rows], label=val_key)
        axis.set_xlabel("epoch")
        axis.set_title(train_key if val_key is None else f"{train_key} vs {val_key}")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_run_name(config: dict[str, Any], role: str) -> str:
    prefix = str(config["mlflow"].get("run_name_prefix", role))
    return f"{prefix}_{timestamp()}"


def expand_physics_stats(stats: dict[str, Any], batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["physics_mean"], dtype=torch.float32, device=device).unsqueeze(0)
    std = torch.tensor(stats["physics_std"], dtype=torch.float32, device=device).unsqueeze(0)
    return (
        mean.expand(batch_size, -1, -1, -1),
        std.expand(batch_size, -1, -1, -1),
    )


def bell_guidance_weight(step_index: int, total_steps: int, w_max: float, peak: float, width: float) -> float:
    if total_steps <= 1:
        return float(w_max)
    progress = step_index / float(total_steps - 1)
    gauss = math.exp(-((progress - peak) ** 2) / (2.0 * width ** 2))
    return float(w_max * gauss)


def polynomial_guidance_weight(step_index: int, total_steps: int, w_min: float, w_max: float, power: float) -> float:
    if total_steps <= 1:
        return float(w_max)
    progress = step_index / float(total_steps - 1)
    return float(w_min + (w_max - w_min) * (progress ** power))


def format_metric(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e3):
        return f"{value:.3e}"
    return f"{value:.5f}"
