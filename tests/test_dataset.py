import json
from pathlib import Path

import pytest

from retrieval_benchmark.dataset import dataset_digest, freeze_test_set, validate_dataset


def _record(identifier: str, split: str, status: str = "verified") -> dict[str, object]:
    return {
        "id": identifier,
        "question": "What does Python do?",
        "category": "factual",
        "difficulty": "easy",
        "split": split,
        "relevant_passages": [{"document_id": "doc", "start": 0, "end": 10, "grade": 2}],
        "provenance": "manual",
        "generation_method": "manual",
        "reviewer": "reviewer",
        "validated_at": "2026-08-16T00:00:00Z",
        "status": status,
    }


def test_validation_rejects_unverified_records(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(_record("q1", "dev", "draft")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verified"):
        validate_dataset(path, require_verified=True)


def test_freeze_is_canonical_and_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "queries.jsonl"
    source.write_text(
        "\n".join(json.dumps(_record(i, "test")) for i in ["q2", "q1"]) + "\n", encoding="utf-8"
    )
    frozen = tmp_path / "test.jsonl"
    manifest = tmp_path / "test.manifest.json"

    digest = freeze_test_set(source, frozen, manifest)

    assert digest == dataset_digest(frozen)
    assert [json.loads(line)["id"] for line in frozen.read_text(encoding="utf-8").splitlines()] == [
        "q1",
        "q2",
    ]
    frozen.write_text(frozen.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(frozen, manifest_path=manifest)
