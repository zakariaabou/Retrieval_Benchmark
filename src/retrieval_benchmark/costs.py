from pydantic import BaseModel, Field


class PricingSnapshot(BaseModel):
    as_of: str
    openai_embedding_per_million_tokens: float = Field(ge=0)
    cohere_rerank_per_1000_searches: float = Field(ge=0)
    local_machine_per_hour: float = Field(default=0, ge=0)


def estimate_hosted_cost(
    pricing: PricingSnapshot, embedding_tokens: int = 0, rerank_searches: int = 0
) -> dict[str, float]:
    embedding = embedding_tokens / 1_000_000 * pricing.openai_embedding_per_million_tokens
    reranking = rerank_searches / 1_000 * pricing.cohere_rerank_per_1000_searches
    return {
        "embedding_usd": round(embedding, 8),
        "reranking_usd": round(reranking, 8),
        "total_usd": round(embedding + reranking, 8),
    }
