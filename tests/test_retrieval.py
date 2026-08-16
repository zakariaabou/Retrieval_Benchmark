import pytest

from retrieval_benchmark.models import SearchResult
from retrieval_benchmark.retrieval import PostgresRetriever, validate_filters, vector_literal


def test_vector_literal_is_numeric_and_finite() -> None:
    assert vector_literal([1.0, -0.5]) == "[1,-0.5]"
    with pytest.raises(ValueError, match="finite"):
        vector_literal([float("nan")])


def test_metadata_filter_allowlist_prevents_sql_identifier_injection() -> None:
    assert validate_filters({"module": "venv"}) == {"module": "venv"}
    with pytest.raises(ValueError, match="Unsupported"):
        validate_filters({"module; DROP TABLE chunks": "x"})


def test_hybrid_and_reranked_search_use_rrf_candidates() -> None:
    import asyncio
    import types

    class FakeReranker:
        name = "fake"

        async def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [0.1, 0.9, 0.2]

    async def run() -> None:
        retriever = PostgresRetriever(object(), "fixed", "hybrid", candidate_depth=3)

        async def lexical(self, query, top_k, filters=None):
            return [
                SearchResult(chunk_id="a", document_id="d", text="A", score=2),
                SearchResult(chunk_id="b", document_id="d", text="B", score=1),
            ]

        async def dense(self, query, top_k, exact, filters=None):
            return [
                SearchResult(chunk_id="b", document_id="d", text="B", score=2),
                SearchResult(chunk_id="c", document_id="d", text="C", score=1),
            ]

        retriever.lexical = types.MethodType(lexical, retriever)
        retriever.dense = types.MethodType(dense, retriever)
        assert [item.chunk_id for item in await retriever.search("q", 3)] == ["b", "a", "c"]

        retriever.strategy = "hybrid_rerank"
        retriever.reranker = FakeReranker()
        assert [item.chunk_id for item in await retriever.search("q", 3)] == ["a", "c", "b"]

    asyncio.run(run())
