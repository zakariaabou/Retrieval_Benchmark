from retrieval_benchmark.costs import PricingSnapshot, estimate_hosted_cost


def test_cost_accounting_uses_versioned_price_snapshot() -> None:
    pricing = PricingSnapshot(
        as_of="2026-08-16",
        openai_embedding_per_million_tokens=0.02,
        cohere_rerank_per_1000_searches=2.0,
    )
    cost = estimate_hosted_cost(pricing, embedding_tokens=500_000, rerank_searches=500)
    assert cost == {"embedding_usd": 0.01, "reranking_usd": 1.0, "total_usd": 1.01}
