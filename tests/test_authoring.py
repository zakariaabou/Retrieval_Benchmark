from retrieval_benchmark.authoring import build_draft_queries
from retrieval_benchmark.models import Document


def test_draft_builder_is_balanced_deterministic_and_never_self_verifies() -> None:
    documents = [
        Document(
            source_uri=f"https://docs.python.org/{index}",
            title=f"Topic {index}",
            text=f"Topic {index} explains a Python capability with enough source text.",
            module=f"module{index}",
        )
        for index in range(5)
    ]
    first = build_draft_queries(documents, total=10, dev_count=5)
    second = build_draft_queries(documents, total=10, dev_count=5)

    assert first == second
    assert len({query.id for query in first}) == 10
    assert sum(query.split == "dev" for query in first) == 5
    assert {query.category for query in first} == {
        "factual",
        "paraphrase",
        "exact_keyword",
        "acronym",
        "multi_passage",
    }
    assert all(query.status == "draft" and query.reviewer is None for query in first)
