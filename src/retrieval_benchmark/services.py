from __future__ import annotations

import time
import uuid
from typing import Protocol

from retrieval_benchmark.models import (
    EvaluateRequest,
    IndexBuildRequest,
    JobRecord,
    SearchRequest,
    SearchResponse,
)


class BenchmarkService(Protocol):
    async def health(self) -> dict[str, object]: ...
    async def enqueue_index(self, request: IndexBuildRequest) -> JobRecord: ...
    async def enqueue_evaluation(self, request: EvaluateRequest) -> JobRecord: ...
    async def get_job(self, job_id: str) -> JobRecord | None: ...
    async def search(self, request: SearchRequest) -> SearchResponse: ...


class InMemoryService:
    """Dependency-free service used for tests and development smoke checks."""

    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}

    async def health(self) -> dict[str, object]:
        return {"status": "ok", "components": {"api": "ok", "database": "not_configured"}}

    async def _enqueue(self, kind: str, payload: dict[str, object]) -> JobRecord:
        job = JobRecord(id=str(uuid.uuid4()), kind=kind, status="queued", payload=payload)
        self.jobs[job.id] = job
        return job

    async def enqueue_index(self, request: IndexBuildRequest) -> JobRecord:
        return await self._enqueue("index_build", request.model_dump())

    async def enqueue_evaluation(self, request: EvaluateRequest) -> JobRecord:
        return await self._enqueue("evaluation", request.model_dump())

    async def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        return SearchResponse(
            results=[],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            configuration_id=request.configuration_id,
        )
