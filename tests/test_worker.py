import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrieval_benchmark.ingestion import write_documents
from retrieval_benchmark.models import Document
from retrieval_benchmark.worker import _build_index, _evaluate


def test_index_build_rejects_wrong_processed_corpus_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed" / "documents.jsonl"
    write_documents(
        [
            Document(
                source_uri="https://docs.python.org/release/3.13.0/library/venv.html",
                title="venv",
                text="Create virtual environments.",
                metadata={"version": "3.13.0"},
            )
        ],
        processed,
    )
    monkeypatch.setattr(
        "retrieval_benchmark.worker.get_settings",
        lambda: SimpleNamespace(data_directory=tmp_path),
    )

    with pytest.raises(ValueError, match="does not match requested version"):
        asyncio.run(
            _build_index(
                object(),
                {
                    "corpus_version": "3.14.6",
                    "chunking": "fixed_256_o32",
                    "embedding_provider": "local",
                    "distance": "cosine",
                },
            )
        )


def test_evaluation_split_override_revalidates_test_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "tuning.yaml").write_text(
        "name: tuning\n"
        "split: dev\n"
        "tuning: true\n"
        "strategy: lexical\n"
        "chunking: fixed_256_o32\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "retrieval_benchmark.worker.get_settings",
        lambda: SimpleNamespace(data_directory=tmp_path / "datasets"),
    )

    with pytest.raises(ValueError, match="test split"):
        asyncio.run(
            _evaluate(
                object(),
                {"configuration": "configs/tuning.yaml", "split": "test"},
            )
        )
