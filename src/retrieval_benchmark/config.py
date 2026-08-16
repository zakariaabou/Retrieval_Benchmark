from pathlib import Path

import yaml

from retrieval_benchmark.models import ExperimentConfig


def load_experiment(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment configuration must be a mapping")
    return ExperimentConfig.model_validate(raw)
