from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from retrieval_benchmark.models import (
    BenchmarkQuery,
    Chunk,
    EvaluateRequest,
    IndexBuildRequest,
    JobRecord,
    SearchRequest,
    SearchResponse,
)
from retrieval_benchmark.providers import Embedder, Reranker
from retrieval_benchmark.retrieval import PostgresRetriever, vector_literal


def create_engine(database_url: str) -> Any:
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise RuntimeError("SQLAlchemy and asyncpg are required for database mode") from exc
    return create_async_engine(database_url, pool_pre_ping=True)


class PostgresJobStore:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def enqueue(self, kind: str, payload: Mapping[str, Any]) -> JobRecord:
        from sqlalchemy import text

        identifier = str(uuid.uuid4())
        statement = text(
            "INSERT INTO jobs (id, kind, status, payload) "
            "VALUES (CAST(:id AS uuid), :kind, 'queued', CAST(:payload AS jsonb))"
        )
        async with self.engine.begin() as connection:
            await connection.execute(
                statement, {"id": identifier, "kind": kind, "payload": json.dumps(dict(payload))}
            )
        return JobRecord(id=identifier, kind=kind, status="queued", payload=dict(payload))

    async def get(self, identifier: str) -> JobRecord | None:
        from sqlalchemy import text

        try:
            uuid.UUID(identifier)
        except ValueError:
            return None

        async with self.engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            "SELECT id::text, kind, status, payload, progress, result, error "
                            "FROM jobs WHERE id = CAST(:id AS uuid)"
                        ),
                        {"id": identifier},
                    )
                )
                .mappings()
                .first()
            )
        return self._record(row) if row else None

    async def claim(self) -> JobRecord | None:
        from sqlalchemy import text

        statement = text(
            "WITH candidate AS (SELECT id FROM jobs WHERE status = 'queued' "
            "ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
            "UPDATE jobs SET status = 'running', started_at = now() FROM candidate "
            "WHERE jobs.id = candidate.id RETURNING jobs.id::text, jobs.kind, jobs.status, "
            "jobs.payload, jobs.progress, jobs.result, jobs.error"
        )
        async with self.engine.begin() as connection:
            row = (await connection.execute(statement)).mappings().first()
        return self._record(row) if row else None

    async def complete(self, identifier: str, result: Mapping[str, Any]) -> None:
        from sqlalchemy import text

        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE jobs SET status='succeeded', progress=1, "
                    "result=CAST(:result AS jsonb), "
                    "finished_at=now() WHERE id=CAST(:id AS uuid)"
                ),
                {"id": identifier, "result": json.dumps(dict(result))},
            )

    async def fail(self, identifier: str, code: str, message: str) -> None:
        from sqlalchemy import text

        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE jobs SET status='failed', error=CAST(:error AS jsonb), "
                    "finished_at=now() "
                    "WHERE id=CAST(:id AS uuid)"
                ),
                {"id": identifier, "error": json.dumps({"code": code, "message": message})},
            )

    @staticmethod
    def _record(row: Mapping[str, Any]) -> JobRecord:
        return JobRecord(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            payload=row["payload"],
            progress=float(row["progress"]),
            result=row["result"],
            error=row["error"],
        )


class ChunkRepository:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        from sqlalchemy import text

        statement = text(
            "INSERT INTO chunks (id, document_id, chunking, content, token_start, token_end, "
            "source_uri, heading, module, metadata) VALUES "
            "(:id, :document_id, :chunking, :content, "
            ":token_start, :token_end, :source_uri, :heading, :module, CAST(:metadata AS jsonb)) "
            "ON CONFLICT (chunking, id) DO UPDATE SET content=EXCLUDED.content, "
            "document_id=EXCLUDED.document_id, token_start=EXCLUDED.token_start, "
            "token_end=EXCLUDED.token_end, source_uri=EXCLUDED.source_uri, "
            "heading=EXCLUDED.heading, module=EXCLUDED.module, metadata=EXCLUDED.metadata"
        )
        values = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunking": chunk.chunking,
                "content": chunk.text,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
                "source_uri": chunk.source_uri,
                "heading": chunk.heading,
                "module": chunk.metadata.get("module"),
                "metadata": json.dumps(chunk.metadata),
            }
            for chunk in chunks
        ]
        async with self.engine.begin() as connection:
            if values:
                await connection.execute(statement, values)
        return len(values)

    async def delete_stale_chunks(self, chunking: str, current_ids: list[str]) -> int:
        """Remove rows from an older corpus build after the replacement is fully embedded."""
        from sqlalchemy import text

        statement = text(
            "DELETE FROM chunks WHERE chunking=:chunking "
            "AND NOT (id = ANY(CAST(:current_ids AS text[])))"
        )
        async with self.engine.begin() as connection:
            result = await connection.execute(
                statement, {"chunking": chunking, "current_ids": current_ids}
            )
        return int(result.rowcount or 0)

    async def store_embeddings(
        self, chunking: str, provider: str, ids: list[str], vectors: list[list[float]]
    ) -> None:
        from sqlalchemy import text

        if len(ids) != len(vectors):
            raise ValueError("chunk and embedding counts differ")
        column = "embedding_openai" if provider == "openai" else "embedding_local"
        sql = f"UPDATE chunks SET {column}=CAST(:vector AS vector) "
        statement = text(sql + "WHERE chunking=:chunking AND id=:id")
        values = [
            {"id": identifier, "chunking": chunking, "vector": vector_literal(vector)}
            for identifier, vector in zip(ids, vectors, strict=True)
        ]
        async with self.engine.begin() as connection:
            if values:
                await connection.execute(statement, values)

    async def project_judgments(
        self, queries: list[BenchmarkQuery], chunking: str
    ) -> list[BenchmarkQuery]:
        """Project verified source token spans onto one deterministic chunk configuration."""
        from sqlalchemy import text

        statement = text(
            "SELECT id FROM chunks WHERE chunking=:chunking AND document_id=:document_id "
            "AND token_start < :passage_end AND token_end > :passage_start ORDER BY id"
        )
        projected: list[BenchmarkQuery] = []
        async with self.engine.connect() as connection:
            for query in queries:
                passages = []
                for passage in query.relevant_passages:
                    rows = await connection.execute(
                        statement,
                        {
                            "chunking": chunking,
                            "document_id": passage.document_id,
                            "passage_start": passage.start,
                            "passage_end": passage.end,
                        },
                    )
                    chunk_ids = [str(row[0]) for row in rows]
                    if not chunk_ids:
                        detail = f"{query.id}/{passage.document_id}"
                        raise ValueError(f"no {chunking} chunk overlaps judgment {detail}")
                    passages.append(passage.model_copy(update={"chunk_ids": chunk_ids}))
                projected.append(query.model_copy(update={"relevant_passages": passages}))
        return projected

    async def rebuild_hnsw(
        self, chunking: str, provider: str, distance: str, m: int, ef_construction: int
    ) -> None:
        from sqlalchemy import text

        partitions = {
            "fixed_128_o16": ("chunks_fixed_128_o16", "c128"),
            "fixed_256_o32": ("chunks_fixed_256_o32", "c256l"),
            "fixed_256_o96": ("chunks_fixed_256_o96", "c256h"),
            "fixed_384_o64": ("chunks_fixed_384_o64", "c384"),
            "structural_256_o32": ("chunks_structural_256_o32", "cstruct"),
        }
        columns = {"local": "embedding_local", "openai": "embedding_openai"}
        opclasses = {
            "cosine": "vector_cosine_ops",
            "inner_product": "vector_ip_ops",
            "l2": "vector_l2_ops",
        }
        if chunking not in partitions or provider not in columns or distance not in opclasses:
            raise ValueError("unsupported HNSW configuration")
        table, prefix = partitions[chunking]
        index = f"{prefix}_{provider}_hnsw"
        async with self.engine.begin() as connection:
            await connection.execute(text(f"DROP INDEX IF EXISTS {index}"))
            await connection.execute(
                text(
                    f"CREATE INDEX {index} ON {table} USING hnsw "
                    f"({columns[provider]} {opclasses[distance]}) WITH "
                    f"(m={int(m)}, ef_construction={int(ef_construction)})"
                )
            )


class PostgresBenchmarkService:
    def __init__(
        self,
        engine: Any,
        chunking: str,
        strategy: str,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.engine = engine
        self.jobs = PostgresJobStore(engine)
        self.retriever = PostgresRetriever(engine, chunking, strategy, embedder, reranker)

    async def health(self) -> dict[str, object]:
        from sqlalchemy import text

        try:
            async with self.engine.connect() as connection:
                version = (
                    await connection.execute(
                        text("SELECT extversion FROM pg_extension WHERE extname='vector'")
                    )
                ).scalar()
            status = "ok" if version else "degraded"
            return {
                "status": status,
                "components": {"api": "ok", "database": "ok", "pgvector": version or "missing"},
            }
        except Exception:
            return {"status": "degraded", "components": {"api": "ok", "database": "unavailable"}}

    async def enqueue_index(self, request: IndexBuildRequest) -> JobRecord:
        return await self.jobs.enqueue("index_build", request.model_dump())

    async def enqueue_evaluation(self, request: EvaluateRequest) -> JobRecord:
        return await self.jobs.enqueue("evaluation", request.model_dump())

    async def get_job(self, job_id: str) -> JobRecord | None:
        return await self.jobs.get(job_id)

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        scoped = PostgresRetriever(
            self.retriever.engine,
            self.retriever.chunking,
            request.strategy,
            self.retriever.embedder,
            self.retriever.reranker,
            self.retriever.rrf_k,
            self.retriever.candidate_depth,
            self.retriever.ef_search,
            self.retriever.distance,
            self.retriever.rerank_candidates,
        )
        results = await scoped.search(request.query, request.top_k, request.filters)
        return SearchResponse(
            results=results,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            configuration_id=request.configuration_id,
        )
