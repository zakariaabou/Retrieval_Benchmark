from __future__ import annotations

import asyncio
import json
from datetime import UTC
from pathlib import Path

import typer

from retrieval_benchmark.dataset import freeze_test_set, validate_dataset
from retrieval_benchmark.ingestion import (
    PythonDocsParser,
    download_corpus,
    extract_zip,
    write_documents,
)
from retrieval_benchmark.reporting import write_report

app = typer.Typer(help="Reproducible production retrieval benchmark")
corpus_app = typer.Typer(help="Download and ingest the corpus")
dataset_app = typer.Typer(help="Validate, review, and freeze evaluation data")
app.add_typer(corpus_app, name="corpus")
app.add_typer(dataset_app, name="dataset")


@corpus_app.command("download")
def corpus_download(
    manifest: Path = Path("datasets/corpus-manifest.json"),
    output: Path = Path("datasets/raw/python-docs.zip"),
) -> None:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    download_corpus(data["url"], data["sha256"], output)
    typer.echo(str(output))


@corpus_app.command("ingest")
def corpus_ingest(
    archive: Path = Path("datasets/raw/python-docs.zip"),
    extracted: Path = Path("datasets/raw/python-docs"),
    output: Path = Path("datasets/processed/documents.jsonl"),
    version: str = "3.14.6",
) -> None:
    extract_zip(archive, extracted)
    roots = [path for path in extracted.iterdir() if path.is_dir()]
    root = roots[0] if len(roots) == 1 else extracted
    documents = PythonDocsParser(version).parse_directory(root)
    write_documents(documents, output)
    typer.echo(f"wrote {len(documents)} documents to {output}")


@dataset_app.command("validate")
def dataset_validate(
    path: Path,
    verified: bool = typer.Option(False, "--verified"),
    manifest: Path | None = None,
) -> None:
    records = validate_dataset(path, require_verified=verified, manifest_path=manifest)
    typer.echo(f"valid: {len(records)} records")


@dataset_app.command("freeze")
def dataset_freeze(
    source: Path = Path("datasets/evaluation/all.jsonl"),
    output: Path = Path("datasets/evaluation/test.jsonl"),
    manifest: Path = Path("datasets/evaluation/test.manifest.json"),
) -> None:
    typer.echo(freeze_test_set(source, output, manifest))


@dataset_app.command("review")
def dataset_review(path: Path = Path("datasets/evaluation/all.jsonl")) -> None:
    """Interactive, append-safe terminal review for draft records."""
    records = validate_dataset(path)
    updated = []
    for record in records:
        if record.status != "draft":
            updated.append(record)
            continue
        typer.echo(f"\n[{record.id}] {record.question}")
        for passage in record.relevant_passages:
            details = (
                f"source={passage.document_id} span={passage.start}:{passage.end} "
                f"grade={passage.grade}"
            )
            typer.echo(f"  {details}")
        action = typer.prompt("approve, reject, or skip", default="skip").strip().lower()
        if action == "approve":
            reviewer = typer.prompt("reviewer")
            from datetime import datetime

            record = record.model_copy(
                update={
                    "status": "verified",
                    "reviewer": reviewer,
                    "validated_at": datetime.now(UTC),
                }
            )
        elif action == "reject":
            record = record.model_copy(update={"status": "rejected"})
        updated.append(record)
    payload = "".join(item.model_dump_json() + "\n" for item in updated)
    temporary = path.with_suffix(".reviewed.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


@dataset_app.command("seed")
def dataset_seed(
    documents: Path = Path("datasets/processed/documents.jsonl"),
    output: Path = Path("datasets/evaluation/all.jsonl"),
) -> None:
    """Create a balanced 300-record draft queue; it never marks records verified."""
    from retrieval_benchmark.authoring import build_draft_queries
    from retrieval_benchmark.ingestion import load_documents

    queries = build_draft_queries(load_documents(documents))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(query.model_dump_json() + "\n" for query in queries), encoding="utf-8"
    )
    typer.echo(f"wrote {len(queries)} draft queries to {output}")


@app.command("report")
def report(results: Path, output: Path = Path("reports/generated")) -> None:
    raw = json.loads(results.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else [raw]
    markdown, html = write_report(rows, output)
    typer.echo(f"{markdown}\n{html}")


@app.command("enqueue-index")
def enqueue_index(config: Path) -> None:
    from retrieval_benchmark.database import PostgresJobStore, create_engine
    from retrieval_benchmark.models import IndexBuildRequest
    from retrieval_benchmark.settings import get_settings

    request = IndexBuildRequest.model_validate_json(config.read_text(encoding="utf-8"))

    async def run() -> None:
        job = await PostgresJobStore(create_engine(get_settings().database_url)).enqueue(
            "index_build", request.model_dump()
        )
        typer.echo(job.id)

    asyncio.run(run())


if __name__ == "__main__":
    app()
