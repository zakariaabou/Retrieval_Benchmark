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


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("top_k: 20\ncandidate_depth: 10\n", "candidate_depth"),
        (
            "strategy: hybrid_rerank\nreranker_provider: local\n"
            "top_k: 20\nrerank_candidates: 10\n",
            "rerank_candidates",
        ),
    ],
)
def test_config_rejects_inconsistent_candidate_depths(
    tmp_path: Path, extra: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    strategy = "" if "strategy:" in extra else "strategy: hybrid\n"
    path.write_text(
        f"name: demo\n{strategy}chunking: fixed_256_o32\n{extra}", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_experiment(path)
