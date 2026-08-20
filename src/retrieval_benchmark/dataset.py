from __future__ import annotations

import hashlib
import json
from pathlib import Path

from retrieval_benchmark.models import BenchmarkQuery


def dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> list[BenchmarkQuery]:
    records: list[BenchmarkQuery] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                records.append(BenchmarkQuery.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid dataset record at line {number}: {exc}") from exc
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate query IDs")
    return records


def validate_dataset(
    path: Path, require_verified: bool = False, manifest_path: Path | None = None
) -> list[BenchmarkQuery]:
    if manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("sha256"), str):
            raise ValueError("dataset manifest is missing a SHA-256 checksum")
        if dataset_digest(path) != manifest["sha256"]:
            raise ValueError("dataset checksum does not match frozen manifest")
    records = _load(path)
    if manifest_path:
        count = manifest.get("count")
        split = manifest.get("split")
        if count is not None and count != len(records):
            raise ValueError("dataset count does not match frozen manifest")
        if split is not None and any(record.split != split for record in records):
            raise ValueError("dataset split does not match frozen manifest")
    if require_verified and any(record.status != "verified" for record in records):
        raise ValueError("all dataset records must be verified")
    return records


def freeze_test_set(source: Path, destination: Path, manifest_path: Path) -> str:
    records = [record for record in validate_dataset(source) if record.split == "test"]
    if not records:
        raise ValueError("source contains no verified test records")
    if any(record.status != "verified" for record in records):
        raise ValueError("all test dataset records must be verified")
    records.sort(key=lambda record: record.id)
    payload = "".join(
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8", newline="\n")
    digest = dataset_digest(destination)
    manifest_path.write_text(
        json.dumps({"sha256": digest, "count": len(records), "split": "test"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return digest
