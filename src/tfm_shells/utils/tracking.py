from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from tfm_shells.config import flatten_for_mlflow


def _resolve_tracking_uri(project_root: Path, raw_uri: str) -> str:
    if raw_uri.startswith("file:"):
        local_path = raw_uri.replace("file:", "", 1).strip()
        if not local_path:
            return (project_root / "mlruns").resolve().as_uri()
        uri_path = Path(local_path)
        if uri_path.is_absolute():
            return uri_path.as_uri()
        return (project_root / uri_path).resolve().as_uri()
    candidate = Path(raw_uri)
    if candidate.is_absolute():
        return candidate.as_uri()
    return (project_root / candidate).resolve().as_uri()


class ExperimentTracker:
    def __init__(self, config: dict[str, Any], project_root: Path, run_name: str) -> None:
        mlflow_cfg = config["mlflow"]
        self.project_root = project_root
        self.tracking_uri = _resolve_tracking_uri(project_root, str(mlflow_cfg["tracking_uri"]))
        self.experiment_name = str(mlflow_cfg["experiment_name"])
        self.run_name = run_name
        self.tags = dict(mlflow_cfg.get("tags", {}))
        self._run = None

    def __enter__(self) -> "ExperimentTracker":
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        mlflow.end_run(status=status)

    @property
    def run_id(self) -> str:
        if self._run is None:
            raise RuntimeError("MLflow run has not been started.")
        return str(self._run.info.run_id)

    def log_config(self, config: dict[str, Any]) -> None:
        flat = flatten_for_mlflow(config)
        for key, value in flat.items():
            if value is None:
                continue
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
