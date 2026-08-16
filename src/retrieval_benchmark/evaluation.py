from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from retrieval_benchmark.metrics import aggregate_metrics, evaluate_ranking
from retrieval_benchmark.models import (
    BenchmarkQuery,
    EvaluationResult,
    QueryEvaluation,
    SearchResult,
)


class Retriever(Protocol):
    async def search(self, query: str, top_k: int) -> list[SearchResult]: ...


class EvaluationRunner:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    async def run(
        self,
        queries: Sequence[BenchmarkQuery],
        top_k: int = 20,
        ks: tuple[int, ...] = (1, 5, 10, 20),
    ) -> EvaluationResult:
        rows: list[QueryEvaluation] = []
        for query in queries:
            started = time.perf_counter()
            results = await self.retriever.search(query.question, top_k)
            elapsed = (time.perf_counter() - started) * 1000
            judgments: dict[str, int] = {}
            for passage in query.relevant_passages:
                for chunk_id in passage.chunk_ids:
                    judgments[chunk_id] = max(judgments.get(chunk_id, 0), passage.grade)
            retrieved = [result.chunk_id for result in results]
            metrics = evaluate_ranking(retrieved, judgments, ks=ks)
            rows.append(
                QueryEvaluation(
                    query_id=query.id,
                    category=query.category,
                    difficulty=query.difficulty,
                    metrics=metrics,
                    latency_ms=elapsed,
                    retrieved_ids=retrieved,
                )
            )
        latencies = [row.latency_ms for row in rows]
        latency = {
            "count": float(len(rows)),
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        }
        return EvaluationResult(
            summary=aggregate_metrics([row.metrics for row in rows]),
            per_query=rows,
            latency=latency,
        )
