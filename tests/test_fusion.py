from retrieval_benchmark.fusion import reciprocal_rank_fusion


def test_rrf_combines_and_tie_breaks_stably() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], rrf_k=60)

    assert [item.id for item in fused] == ["b", "a", "c"]
    assert fused[0].component_ranks == [2, 1]
    assert fused[1].score > fused[2].score


def test_rrf_rejects_invalid_constant() -> None:
    try:
        reciprocal_rank_fusion([["a"]], rrf_k=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rrf_uses_first_occurrence_of_duplicate_result() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "a"]])

    assert fused[0].id == "a"
    assert fused[0].component_ranks == [1]
