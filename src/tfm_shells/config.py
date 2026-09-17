from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must contain a dictionary.")
    config = deepcopy(data)
    config["_meta"] = {
        "config_path": str(path),
        "project_root": str(path.parent.parent.resolve()),
        "config_name": path.stem,
    }
    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    sanitized = deepcopy(config)
    sanitized.pop("_meta", None)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sanitized, handle, sort_keys=False)


def project_root(config: dict[str, Any]) -> Path:
    return Path(config["_meta"]["project_root"]).resolve()


def config_name(config: dict[str, Any]) -> str:
    return str(config["_meta"]["config_name"])


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root(config) / path).resolve()


def flatten_for_mlflow(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if key == "_meta":
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_for_mlflow(value, prefix=full_key))
        elif isinstance(value, list):
            flat[full_key] = ",".join(str(item) for item in value)
        else:
            flat[full_key] = value
    return flat
