from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

from retrieval_benchmark.fusion import reciprocal_rank_fusion
from retrieval_benchmark.models import SearchResult
from retrieval_benchmark.providers import Embedder, Reranker

ALLOWED_FILTERS = {"module", "heading", "source_uri", "document_id"}


def validate_filters(filters: Mapping[str, str]) -> dict[str, str]:
    unsupported = sorted(set(filters) - ALLOWED_FILTERS)
    if unsupported:
        raise ValueError(f"Unsupported metadata filters: {', '.join(unsupported)}")
    return dict(filters)


def vector_literal(vector: Sequence[float]) -> str:
    values = []
    for item in vector:
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("vector values must be finite")
        values.append(format(number, ".9g"))
    return "[" + ",".join(values) + "]"


class PostgresRetriever:
    def __init__(
        self,
        engine: AsyncEngine,
        chunking: str,
        strategy: str = "hybrid",
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_depth: int = 100,
        ef_search: int = 40,
        distance: str = "cosine",
    ) -> None:
        self.engine = engine
        self.chunking = chunking
        self.strategy = strategy
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth
        self.ef_search = ef_search
        self.distance = distance

    @staticmethod
    def _filter_sql(filters: Mapping[str, str]) -> tuple[str, dict[str, str]]:
        checked = validate_filters(filters)
        clauses = [f"{key} = :filter_{key}" for key in checked]
        return (" AND " + " AND ".join(clauses) if clauses else ""), {
            f"filter_{key}": value for key, value in checked.items()
        }

    async def lexical(
        self, query: str, top_k: int, filters: Mapping[str, str] | None = None
    ) -> list[SearchResult]:
        from sqlalchemy import text

        filter_sql, filter_parameters = self._filter_sql(filters or {})
        parameters: dict[str, object] = dict(filter_parameters)
        statement = text(
            "SELECT id, document_id, content, source_uri, heading, "
            "ts_rank_cd(search_vector, websearch_to_tsquery('english', :query)) AS score "
            "FROM chunks WHERE chunking = :chunking "
            "AND search_vector @@ websearch_to_tsquery('english', :query)"
            f"{filter_sql} ORDER BY score DESC, id LIMIT :limit"
        )
        parameters.update(query=query, chunking=self.chunking, limit=top_k)
        async with self.engine.connect() as connection:
            rows = (await connection.execute(statement, parameters)).mappings().all()
        return [self._row(dict(row)) for row in rows]

    async def dense(
        self, query: str, top_k: int, exact: bool, filters: Mapping[str, str] | None = None
    ) -> list[SearchResult]:
        from sqlalchemy import text

        if self.embedder is None:
            raise RuntimeError("dense retrieval requires an embedder")
        vector = vector_literal((await self.embedder.embed([query]))[0])
        column = "embedding_openai" if self.embedder.dimensions == 1536 else "embedding_local"
        operators = {
            "cosine": ("<=>", "1 -"),
            "inner_product": ("<#>", "-"),
            "l2": ("<->", "-"),
        }
        if self.distance not in operators:
            raise ValueError(f"unsupported distance: {self.distance}")
        operator, score_prefix = operators[self.distance]
        filter_sql, filter_parameters = self._filter_sql(filters or {})
        parameters: dict[str, object] = dict(filter_parameters)
        select_score = (
            f"SELECT id, document_id, content, source_uri, heading, "
            f"{score_prefix} ({column} {operator} :vector) AS score "
        )
        statement = text(
            select_score
            + f"FROM chunks WHERE chunking = :chunking AND {column} IS NOT NULL{filter_sql} "
            f"ORDER BY {column} {operator} :vector, id LIMIT :limit"
        )
        parameters.update(vector=vector, chunking=self.chunking, limit=top_k)
        async with self.engine.begin() as connection:
            if exact:
                await connection.execute(text("SET LOCAL enable_indexscan = off"))
                await connection.execute(text("SET LOCAL enable_bitmapscan = off"))
            else:
                await connection.execute(
                    text("SELECT set_config('hnsw.ef_search', :value, true)"),
                    {"value": str(self.ef_search)},
                )
            rows = (await connection.execute(statement, parameters)).mappings().all()
        return [self._row(dict(row)) for row in rows]

    async def search(
        self, query: str, top_k: int, filters: Mapping[str, str] | None = None
    ) -> list[SearchResult]:
        if self.strategy == "lexical":
            return await self.lexical(query, top_k, filters)
        if self.strategy in {"dense_exact", "dense_hnsw"}:
            return await self.dense(query, top_k, self.strategy == "dense_exact", filters)
        lexical = await self.lexical(query, self.candidate_depth, filters)
        dense = await self.dense(query, self.candidate_depth, False, filters)
        lookup = {item.chunk_id: item for item in [*lexical, *dense]}
        fused = reciprocal_rank_fusion(
            [[item.chunk_id for item in lexical], [item.chunk_id for item in dense]], self.rrf_k
        )
        results = [
            lookup[item.id].model_copy(
                update={
                    "score": item.score,
                    "component_scores": {
                        "lexical_rank": float(item.component_ranks[0] or 0),
                        "dense_rank": float(item.component_ranks[1] or 0),
                    },
                }
            )
            for item in fused[: self.candidate_depth]
        ]
        if self.strategy == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("hybrid_rerank requires a reranker")
            scores = await self.reranker.rerank(query, [item.text for item in results])
            results = sorted(
                [
                    item.model_copy(update={"score": score})
                    for item, score in zip(results, scores, strict=True)
                ],
                key=lambda item: (-item.score, item.chunk_id),
            )
        return results[:top_k]

    @staticmethod
    def _row(row: Mapping[str, Any]) -> SearchResult:
        return SearchResult(
            chunk_id=row["id"],
            document_id=row["document_id"],
            text=row["content"],
            score=float(row["score"]),
            source_uri=row.get("source_uri"),
            heading=row.get("heading"),
        )
