import pytest

from retrieval_benchmark.metrics import aggregate_metrics, evaluate_ranking


def test_known_ranking_metrics() -> None:
    result = evaluate_ranking(["a", "b", "c"], {"b": 2, "c": 1}, ks=(1, 2, 3))

    assert result["recall@1"] == 0
    assert result["recall@2"] == pytest.approx(0.5)
    assert result["precision@2"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
    assert 0 < result["ndcg@3"] <= 1
    assert result["context_recall@3"] == 1


def test_empty_judgments_are_safe() -> None:
    result = evaluate_ranking(["a"], {}, ks=(1,))
    assert result == {
        "recall@1": 0.0,
        "precision@1": 0.0,
        "ndcg@1": 0.0,
        "context_precision@1": 0.0,
        "context_recall@1": 0.0,
        "mrr": 0.0,
    }


def test_duplicate_results_cannot_inflate_recall() -> None:
    result = evaluate_ranking(["a", "a", "b"], {"a": 1, "b": 1}, ks=(2,))

    assert result["recall@2"] == 1.0
    assert result["precision@2"] == 1.0


def test_aggregate_metrics_computes_means() -> None:
    assert aggregate_metrics([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}
