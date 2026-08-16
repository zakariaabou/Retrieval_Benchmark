from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, cast


class Embedder(Protocol):
    name: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class HashEmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, list[float]] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(provider: str, text: str) -> str:
        return hashlib.sha256(f"{provider}\x1f{text}".encode()).hexdigest()

    def get(self, provider: str, text: str) -> list[float] | None:
        return self._data.get(self.key(provider, text))

    def set(self, provider: str, text: str, value: list[float]) -> None:
        self._data[self.key(provider, text)] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)


class CachedEmbedder:
    def __init__(self, provider: Embedder, cache: HashEmbeddingCache) -> None:
        self.provider = provider
        self.cache = cache
        self.name = provider.name
        self.dimensions = provider.dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        output: list[list[float] | None] = [self.cache.get(self.name, text) for text in texts]
        missing_positions = [index for index, item in enumerate(output) if item is None]
        if missing_positions:
            missing = [texts[index] for index in missing_positions]
            generated = await self.provider.embed(missing)
            if len(generated) != len(missing):
                raise RuntimeError("embedding provider returned an unexpected result count")
            for index, vector in zip(missing_positions, generated, strict=True):
                output[index] = vector
                self.cache.set(self.name, texts[index], vector)
        return [item for item in output if item is not None]


class LocalBGEEmbedder:
    name = "BAAI/bge-small-en-v1.5@main"
    dimensions = 384

    def __init__(self, revision: str = "main") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the local-models extra") from exc
        self.name = f"BAAI/bge-small-en-v1.5@{revision}"
        self._model = SentenceTransformer("BAAI/bge-small-en-v1.5", revision=revision)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return cast(list[list[float]], vectors.tolist())


class OpenAIEmbedder:
    name = "openai:text-embedding-3-small"
    dimensions = 1536

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("install the hosted extra") from exc
        self._client = AsyncOpenAI(api_key=api_key)
        self.usage_tokens = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model="text-embedding-3-small", input=texts)
        self.usage_tokens += response.usage.total_tokens
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class LocalCrossEncoderReranker:
    name = "cross-encoder/ms-marco-MiniLM-L6-v2@main"

    def __init__(self, revision: str = "main") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("install the local-models extra") from exc
        self.name = f"cross-encoder/ms-marco-MiniLM-L6-v2@{revision}"
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", revision=revision)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        scores = self._model.predict([(query, document) for document in documents]).tolist()
        return cast(list[float], scores)


class CohereReranker:
    name = "cohere:rerank-v4.0-pro"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from cohere import AsyncClientV2
        except ImportError as exc:
            raise RuntimeError("install the hosted extra") from exc
        self._client = AsyncClientV2(api_key=api_key)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = await self._client.rerank(
            model="rerank-v4.0-pro", query=query, documents=documents, top_n=len(documents)
        )
        scores = [0.0] * len(documents)
        for result in response.results:
            scores[result.index] = result.relevance_score
        return scores
