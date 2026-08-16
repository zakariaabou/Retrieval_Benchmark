from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from retrieval_benchmark.chunking import DEFAULT_CHUNKING_CONFIGS, Chunker
from retrieval_benchmark.config import load_experiment
from retrieval_benchmark.database import ChunkRepository, PostgresJobStore, create_engine
from retrieval_benchmark.dataset import dataset_digest, validate_dataset
from retrieval_benchmark.evaluation import EvaluationRunner
from retrieval_benchmark.ingestion import load_documents
from retrieval_benchmark.providers import (
    CachedEmbedder,
    CohereReranker,
    HashEmbeddingCache,
    LocalBGEEmbedder,
    LocalCrossEncoderReranker,
    OpenAIEmbedder,
    Reranker,
)
from retrieval_benchmark.reporting import write_report
from retrieval_benchmark.retrieval import PostgresRetriever
from retrieval_benchmark.settings import get_settings
from retrieval_benchmark.tracking import log_evaluation, reproducibility_metadata

LOGGER = logging.getLogger(__name__)


def _embedder(provider: str) -> CachedEmbedder:
    settings = get_settings()
    implementation = (
        OpenAIEmbedder(settings.openai_api_key) if provider == "openai" else LocalBGEEmbedder()
    )
    return CachedEmbedder(
        implementation, HashEmbeddingCache(settings.cache_directory / f"{provider}-embeddings.json")
    )


async def _build_index(engine: Any, payload: dict[str, object]) -> dict[str, object]:
    settings = get_settings()
    documents = load_documents(settings.data_directory / "processed" / "documents.jsonl")
    config_name = str(payload["chunking"])
    config = next((item for item in DEFAULT_CHUNKING_CONFIGS if item.name == config_name), None)
    if config is None:
        raise ValueError(f"unknown chunking configuration: {config_name}")
    chunks = [chunk for document in documents for chunk in Chunker().chunk(document, config)]
    repository = ChunkRepository(engine)
    await repository.upsert_chunks(chunks)
    provider_name = str(payload["embedding_provider"])
    embedder = _embedder(provider_name)
    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        vectors = await embedder.embed([chunk.text for chunk in batch])
        await repository.store_embeddings(
            config_name, provider_name, [chunk.id for chunk in batch], vectors
        )
    await repository.rebuild_hnsw(
        config_name,
        provider_name,
        str(payload["distance"]),
        int(str(payload.get("hnsw_m", 16))),
        int(str(payload.get("ef_construction", 64))),
    )
    return {"chunks": len(chunks), "documents": len(documents), "embedding_provider": provider_name}


async def _evaluate(engine: Any, payload: dict[str, object]) -> dict[str, object]:
    settings = get_settings()
    project_root = Path.cwd().resolve()
    config_path = (project_root / str(payload["configuration"])).resolve()
    configs_root = (project_root / "configs").resolve()
    if configs_root not in config_path.parents or config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("configuration path is outside the configs directory")
    config = load_experiment(config_path)
    requested_split = str(payload.get("split", config.split))
    if config.split != requested_split:
        config = config.model_copy(update={"split": requested_split})
    dataset_path = settings.data_directory / "evaluation" / f"{config.split}.jsonl"
    manifest = (
        settings.data_directory / "evaluation" / "test.manifest.json"
        if config.split == "test"
        else None
    )
    queries = validate_dataset(dataset_path, require_verified=True, manifest_path=manifest)
    queries = await ChunkRepository(engine).project_judgments(queries, config.chunking)
    embedder = _embedder(config.embedding_provider) if config.strategy != "lexical" else None
    reranker: Reranker | None = None
    if config.reranker_provider == "local":
        reranker = LocalCrossEncoderReranker()
    elif config.reranker_provider == "cohere":
        reranker = CohereReranker(settings.cohere_api_key)
    retriever = PostgresRetriever(
        engine,
        config.chunking,
        config.strategy,
        embedder=embedder,
        reranker=reranker,
        rrf_k=config.rrf_k,
        candidate_depth=config.candidate_depth,
        distance=config.distance,
    )
    result = await EvaluationRunner(retriever).run(queries, top_k=config.top_k)
    output = settings.reports_directory / config.name
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "evaluation.json"
    results_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    row = {
        "name": config.name,
        **result.summary,
        "p95_ms": result.latency["p95_ms"],
        "cost_usd": 0.0,
    }
    markdown, html = write_report([row], output)
    metadata = reproducibility_metadata(
        "see-corpus-manifest", dataset_digest(dataset_path), config.model_dump()
    )
    try:
        run_id = log_evaluation(
            "retrieval-benchmark",
            config.name,
            metadata,
            result.summary,
            [results_path, markdown, html],
            settings.mlflow_tracking_uri,
        )
    except RuntimeError:
        run_id = "mlflow-not-installed"
    return {
        "run_id": run_id,
        "summary": result.summary,
        "artifacts": [str(results_path), str(markdown), str(html)],
    }


async def handle_job(
    store: PostgresJobStore, engine: Any, job_id: str, kind: str, payload: dict[str, object]
) -> None:
    try:
        if kind == "index_build":
            await store.complete(job_id, await _build_index(engine, payload))
        elif kind == "evaluation":
            await store.complete(job_id, await _evaluate(engine, payload))
        else:
            await store.fail(job_id, "unknown_job", f"Unsupported job type: {kind}")
    except Exception:
        LOGGER.exception("job %s failed", job_id)
        await store.fail(job_id, "job_failed", "Job execution failed; inspect worker logs")


async def run_worker() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    store = PostgresJobStore(engine)
    while True:
        job = await store.claim()
        if job:
            await handle_job(store, engine, job.id, job.kind, job.payload)
        else:
            await asyncio.sleep(settings.worker_poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
