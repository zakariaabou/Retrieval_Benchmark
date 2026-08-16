import asyncio

from retrieval_benchmark.evaluation import EvaluationRunner
from retrieval_benchmark.models import BenchmarkQuery, PassageJudgment, SearchResult


class FakeRetriever:
    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        return [SearchResult(chunk_id="c1", document_id="doc", text="answer", score=1.0)]


def test_runner_produces_per_query_and_summary_metrics() -> None:
    query = BenchmarkQuery(
        id="q1",
        question="Question?",
        category="factual",
        difficulty="easy",
        split="dev",
        relevant_passages=[
            PassageJudgment(document_id="doc", chunk_ids=["c1"], start=0, end=5, grade=2)
        ],
        provenance="manual",
        generation_method="manual",
        reviewer="human",
        validated_at="2026-08-16T00:00:00Z",
        status="verified",
    )

    result = asyncio.run(EvaluationRunner(FakeRetriever()).run([query], top_k=5))

    assert result.summary["recall@1"] == 1
    assert result.per_query[0].query_id == "q1"
    assert result.latency["count"] == 1
