from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def reproducibility_metadata(
    corpus_sha256: str, dataset_sha256: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = "uncommitted"
    return {
        "git_sha": git_sha,
        "corpus_sha256": corpus_sha256,
        "dataset_sha256": dataset_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": dict(config),
    }


def log_evaluation(
    experiment: str,
    run_name: str,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, float],
    artifacts: list[Path],
    tracking_uri: str,
) -> str:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("install the tracking extra to use MLflow") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        flattened = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        mlflow.log_params(flattened)
        mlflow.log_metrics(dict(metrics))
        metadata_path = Path(".cache/retrieval-benchmark/run-metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(metadata_path))
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return str(run.info.run_id)
