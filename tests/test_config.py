from pathlib import Path

import pytest

from retrieval_benchmark.config import load_experiment


def test_config_loads_and_forbids_test_tuning(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: demo\nsplit: test\ntuning: true\nstrategy: hybrid\nchunking: fixed_256_o32\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test split"):
        load_experiment(path)
