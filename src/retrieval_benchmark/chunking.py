from __future__ import annotations

import hashlib

from retrieval_benchmark.models import Chunk, ChunkingConfig, Document


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


class Chunker:
    """Deterministic whitespace-token chunker with source token offsets."""

    def chunk(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        if config.mode == "structural" and document.sections:
            return self._structural(document, config)
        return self._fixed(document, config, document.text.split(), heading=None, base_offset=0)

    def _fixed(
        self,
        document: Document,
        config: ChunkingConfig,
        tokens: list[str],
        heading: str | None,
        base_offset: int,
    ) -> list[Chunk]:
        if not tokens:
            return []
        chunks: list[Chunk] = []
        step = config.max_tokens - config.overlap_tokens
        for index, start in enumerate(range(0, len(tokens), step)):
            end = min(start + config.max_tokens, len(tokens))
            text = " ".join(tokens[start:end])
            chunks.append(
                Chunk(
                    id=stable_id(document.id, config.name, str(index + base_offset), text),
                    document_id=document.id,
                    chunking=config.name,
                    text=text,
                    token_start=base_offset + start,
                    token_end=base_offset + end,
                    source_uri=document.source_uri,
                    heading=heading,
                    metadata=document.metadata,
                )
            )
            if end == len(tokens):
                break
        return chunks

    def _structural(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = 0
        ordinal = 0
        for section in document.sections:
            tokens = section.get("text", "").split()
            section_chunks = self._fixed(
                document, config, tokens, section.get("heading"), base_offset=offset
            )
            for chunk in section_chunks:
                chunks.append(
                    chunk.model_copy(
                        update={"id": stable_id(document.id, config.name, str(ordinal), chunk.text)}
                    )
                )
                ordinal += 1
            offset += len(tokens)
        return chunks


DEFAULT_CHUNKING_CONFIGS = [
    ChunkingConfig(
        name="fixed_128_o16", mode="fixed", target_tokens=128, max_tokens=128, overlap_tokens=16
    ),
    ChunkingConfig(
        name="fixed_256_o32", mode="fixed", target_tokens=256, max_tokens=256, overlap_tokens=32
    ),
    ChunkingConfig(
        name="fixed_256_o96", mode="fixed", target_tokens=256, max_tokens=256, overlap_tokens=96
    ),
    ChunkingConfig(
        name="fixed_384_o64", mode="fixed", target_tokens=384, max_tokens=384, overlap_tokens=64
    ),
    ChunkingConfig(
        name="structural_256_o32",
        mode="structural",
        target_tokens=256,
        max_tokens=384,
        overlap_tokens=32,
    ),
]
