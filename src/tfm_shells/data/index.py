from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def build_dataset_index(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*.npz"))
    records: list[dict[str, Any]] = []

    for path in tqdm(files, desc="Indexing dataset", leave=False):
        with np.load(path) as data:
            mf_mean = float(data["mf"].mean())
            fz_mean = float(data["fz"].mean())
            z_min = float(data["z"].min())
            z_max = float(data["z"].max())
        is_hole = "shell_hole" in path.name
        records.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "subset": "hole" if is_hole else "solid",
                "mf_mean": mf_mean,
                "fz_mean": fz_mean,
                "z_min": z_min,
                "z_max": z_max,
            }
        )
    return records


def filter_records(
    records: list[dict[str, Any]],
    subset: str,
    min_mf_mean: float | None = None,
) -> list[dict[str, Any]]:
    if subset not in {"solid", "hole", "all"}:
        raise ValueError(f"Unsupported subset: {subset}")

    filtered = records
    if subset != "all":
        filtered = [record for record in filtered if record["subset"] == subset]
    if min_mf_mean is not None:
        filtered = [record for record in filtered if record["mf_mean"] >= float(min_mf_mean)]
    if not filtered:
        raise RuntimeError("Dataset filtering returned zero records.")
    return filtered


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    if len(records) < 2:
        raise RuntimeError("At least two records are required to split the dataset.")

    stratify = None
    subsets = {record["subset"] for record in records}
    if len(subsets) > 1:
        stratify = [record["subset"] for record in records]

    train_records, val_records = train_test_split(
        records,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_records), list(val_records)


def dataset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    mf_values = np.array([record["mf_mean"] for record in records], dtype=np.float64)
    subsets: dict[str, int] = {}
    for record in records:
        subsets[record["subset"]] = subsets.get(record["subset"], 0) + 1
    return {
        "count": len(records),
        "subset_counts": subsets,
        "mf_mean": float(mf_values.mean()),
        "mf_std": float(mf_values.std()),
        "mf_min": float(mf_values.min()),
        "mf_max": float(mf_values.max()),
    }
