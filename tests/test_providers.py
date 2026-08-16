import asyncio

from retrieval_benchmark.providers import CachedEmbedder, HashEmbeddingCache


class FakeEmbedder:
    name = "fake"
    dimensions = 2

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 0.0] for text in texts]


def test_embedding_cache_avoids_duplicate_provider_calls(tmp_path) -> None:
    async def run() -> None:
        provider = FakeEmbedder()
        cached = CachedEmbedder(provider, HashEmbeddingCache(tmp_path / "cache.json"))
        assert await cached.embed(["hello"]) == [[5.0, 0.0]]
        assert await cached.embed(["hello"]) == [[5.0, 0.0]]
        assert provider.calls == 1

    asyncio.run(run())
