from __future__ import annotations

import random
from typing import Any

import torch


TYPE_TO_ID = {"solid": 0, "hole": 1}
ID_TO_TYPE = {value: key for key, value in TYPE_TO_ID.items()}


def type_id_from_subset(subset: str) -> int:
    key = str(subset).lower()
    if key not in TYPE_TO_ID:
        raise ValueError(f"Unsupported structural type: {subset}")
    return TYPE_TO_ID[key]


def type_channel_value(type_id: int) -> float:
    if int(type_id) not in ID_TO_TYPE:
        raise ValueError(f"Unsupported structural type id: {type_id}")
    return -1.0 if int(type_id) == TYPE_TO_ID["solid"] else 1.0


def parse_condition_type(raw_type: str | int) -> tuple[int, str]:
    if isinstance(raw_type, str):
        lowered = raw_type.strip().lower()
        if lowered in TYPE_TO_ID:
            type_id = TYPE_TO_ID[lowered]
            return type_id, ID_TO_TYPE[type_id]
        if lowered.isdigit():
            raw_type = int(lowered)
        else:
            raise ValueError(f"Unsupported conditioning type: {raw_type}")

    type_id = int(raw_type)
    if type_id not in ID_TO_TYPE:
        raise ValueError(f"Unsupported conditioning type id: {raw_type}")
    return type_id, ID_TO_TYPE[type_id]


def type_channel_tensor(
    batch_size: int,
    height: int,
    width: int,
    type_id: int,
    device: torch.device,
) -> torch.Tensor:
    value = type_channel_value(type_id)
    return torch.full((batch_size, 1, height, width), value, dtype=torch.float32, device=device)


def split_records_balanced_validation(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    grouped = {subset: [record for record in records if record["subset"] == subset] for subset in TYPE_TO_ID}
    missing = [subset for subset, subset_records in grouped.items() if not subset_records]
    if missing:
        raise RuntimeError(
            "Balanced conditional validation requires both solid and hole records. "
            f"Missing: {', '.join(missing)}"
        )

    min_count = min(len(subset_records) for subset_records in grouped.values())
    val_per_type = max(int(round(min_count * val_ratio)), 1)
    if val_per_type >= min_count:
        val_per_type = min_count - 1
    if val_per_type < 1:
        raise RuntimeError("Not enough records per structural type to build a balanced validation split.")

    rng = random.Random(seed)
    val_names: set[str] = set()
    for subset_records in grouped.values():
        shuffled = list(subset_records)
        rng.shuffle(shuffled)
        val_names.update(str(record["name"]) for record in shuffled[:val_per_type])

    train_records = [record for record in records if str(record["name"]) not in val_names]
    val_records = [record for record in records if str(record["name"]) in val_names]
    return train_records, val_records
