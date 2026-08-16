from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def content_id(*parts: str, length: int = 20) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    source_uri: str
    title: str
    text: str
    module: str | None = None
    sections: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_id(self) -> Document:
        if not self.id:
            self.id = content_id(self.source_uri, self.title, self.text)
        return self


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mode: Literal["fixed", "structural"]
    target_tokens: int = Field(gt=0, le=512)
    max_tokens: int = Field(gt=0, le=512)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ChunkingConfig:
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    document_id: str
    chunking: str
    text: str
    token_start: int
    token_end: int
    source_uri: str
    heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PassageJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    grade: int = Field(ge=1, le=3)
    chunk_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_span(self) -> PassageJudgment:
        if self.end <= self.start:
            raise ValueError("passage end must be greater than start")
        return self


class BenchmarkQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str = Field(min_length=3)
    category: Literal["factual", "paraphrase", "exact_keyword", "acronym", "multi_passage"]
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["dev", "test"]
    relevant_passages: list[PassageJudgment] = Field(min_length=1)
    reference_answer: str | None = None
    provenance: str
    generation_method: str
    reviewer: str | None = None
    validated_at: datetime | None = None
    status: Literal["draft", "verified", "rejected"] = "draft"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    split: Literal["dev", "test"] = "dev"
    tuning: bool = False
    strategy: Literal["lexical", "dense_exact", "dense_hnsw", "hybrid", "hybrid_rerank"]
    chunking: str
    embedding_provider: Literal["local", "openai"] = "local"
    distance: Literal["cosine", "inner_product", "l2"] = "cosine"
    reranker_provider: Literal["none", "local", "cohere"] = "none"
    top_k: int = Field(default=20, gt=0, le=200)
    candidate_depth: int = Field(default=100, gt=0, le=1000)
    rrf_k: int = Field(default=60, gt=0)
    rerank_candidates: int = Field(default=50, gt=0, le=200)
    seed: int = 42

    @model_validator(mode="after")
    def protect_test_split(self) -> ExperimentConfig:
        if self.split == "test" and self.tuning:
            raise ValueError("tuning is forbidden on the test split")
        if self.strategy == "hybrid_rerank" and self.reranker_provider == "none":
            raise ValueError("hybrid_rerank requires a reranker_provider")
        return self


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source_uri: str | None = None
    heading: str | None = None
    component_scores: dict[str, float] = Field(default_factory=dict)


class QueryEvaluation(BaseModel):
    query_id: str
    category: str
    difficulty: str
    metrics: dict[str, float]
    latency_ms: float
    retrieved_ids: list[str]


class EvaluationResult(BaseModel):
    summary: dict[str, float]
    per_query: list[QueryEvaluation]
    latency: dict[str, float]


class IndexBuildRequest(BaseModel):
    corpus_version: str
    chunking: str
    embedding_provider: Literal["local", "openai"]
    distance: Literal["cosine", "inner_product", "l2"] = "cosine"
    hnsw_m: int = Field(default=16, ge=4, le=64)
    ef_construction: int = Field(default=64, ge=8, le=512)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    configuration_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=200)
    strategy: Literal["lexical", "dense_exact", "dense_hnsw", "hybrid", "hybrid_rerank"] = "hybrid"
    filters: dict[str, str] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    configuration: str = Field(min_length=1, max_length=300)
    split: Literal["dev", "test"] = "dev"

    @model_validator(mode="after")
    def safe_configuration_path(self) -> EvaluateRequest:
        path = PurePosixPath(self.configuration.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("configuration must be a safe relative path")
        if path.parts[0] != "configs" or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("configuration must reference a YAML file under configs")
        self.configuration = path.as_posix()
        return self


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobRecord(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    payload: dict[str, Any]
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    elapsed_ms: float
    configuration_id: str | None = None
