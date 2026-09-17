from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tfm_shells.data.conditioning import type_channel_value, type_id_from_subset
from tfm_shells.constants import PHYSICS_KEYS


def _normalize_minmax(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scale = maximum - minimum
    if abs(scale) < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (2.0 * ((array - minimum) / scale) - 1.0).astype(np.float32)


def _physics_stack(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.concatenate([data[key].astype(np.float32) for key in PHYSICS_KEYS], axis=0)


def compute_normalization_stats(records: list[dict[str, Any]], include_physics: bool) -> dict[str, Any]:
    z_mins: list[float] = []
    z_maxs: list[float] = []
    fz_mins: list[float] = []
    fz_maxs: list[float] = []

    p_sum = np.zeros(len(PHYSICS_KEYS), dtype=np.float64)
    p_sq_sum = np.zeros(len(PHYSICS_KEYS), dtype=np.float64)
    pixel_count = 0

    for record in records:
        with np.load(record["path"]) as data:
            z = data["z"].astype(np.float64)
            fz = data["fz"].astype(np.float64)
            z_mins.append(float(z.min()))
            z_maxs.append(float(z.max()))
            fz_mins.append(float(fz.min()))
            fz_maxs.append(float(fz.max()))

            if include_physics:
                for index, key in enumerate(PHYSICS_KEYS):
                    arr = data[key].astype(np.float64)
                    p_sum[index] += arr.sum()
                    p_sq_sum[index] += np.square(arr).sum()
                pixel_count += int(np.prod(data["z"].shape[1:]))

    stats: dict[str, Any] = {
        "z_min": min(z_mins),
        "z_max": max(z_maxs),
        "fz_min": min(fz_mins),
        "fz_max": max(fz_maxs),
    }

    if include_physics:
        mean = (p_sum / pixel_count).reshape(len(PHYSICS_KEYS), 1, 1)
        variance = np.maximum((p_sq_sum / pixel_count) - np.square(p_sum / pixel_count), 1e-12)
        std = np.sqrt(variance).reshape(len(PHYSICS_KEYS), 1, 1)
        stats["physics_mean"] = mean.astype(np.float32).tolist()
        stats["physics_std"] = std.astype(np.float32).tolist()
    return stats


class ShellDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        stats: dict[str, Any],
        include_physics: bool,
    ) -> None:
        self.records = records
        self.stats = stats
        self.include_physics = include_physics
        self.z_min = float(stats["z_min"])
        self.z_max = float(stats["z_max"])
        self.fz_min = float(stats["fz_min"])
        self.fz_max = float(stats["fz_max"])
        self.physics_mean = None
        self.physics_std = None
        if include_physics:
            self.physics_mean = np.asarray(stats["physics_mean"], dtype=np.float32)
            self.physics_std = np.asarray(stats["physics_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        path = Path(record["path"])
        with np.load(path) as data:
            z = data["z"].astype(np.float32)
            fz = data["fz"].astype(np.float32)
            z_norm = _normalize_minmax(z, self.z_min, self.z_max)
            fz_norm = _normalize_minmax(fz, self.fz_min, self.fz_max)
            type_id = type_id_from_subset(str(record["subset"]))
            z_tensor = torch.from_numpy(z_norm)

            item: dict[str, torch.Tensor | str] = {
                "name": path.name,
                "z": z_tensor,
                "fz_norm": torch.from_numpy(fz_norm),
                "fz_real": torch.from_numpy(fz),
                "type_id": torch.tensor(type_id, dtype=torch.long),
                "type_channel": torch.full_like(z_tensor, type_channel_value(type_id)),
            }

            if self.include_physics and self.physics_mean is not None and self.physics_std is not None:
                physics = _physics_stack(data)
                physics_norm = ((physics - self.physics_mean) / self.physics_std).astype(np.float32)
                item.update(
                    {
                        "physics": torch.from_numpy(physics_norm),
                        "ds": torch.from_numpy(data["ds"].astype(np.float32)),
                        "dv": torch.from_numpy(data["dv"].astype(np.float32)),
                        "mf_true": torch.from_numpy(data["mf"].astype(np.float32)),
                    }
                )
            return item
