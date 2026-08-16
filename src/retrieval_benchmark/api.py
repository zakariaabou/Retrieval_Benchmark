from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from retrieval_benchmark.models import (
    EvaluateRequest,
    IndexBuildRequest,
    JobRecord,
    JobResponse,
    SearchRequest,
    SearchResponse,
)
from retrieval_benchmark.services import BenchmarkService, InMemoryService


def create_app(service: BenchmarkService | None = None) -> FastAPI:
    application = FastAPI(title="Retrieval Benchmark API", version="0.1.0")
    backend = service or InMemoryService()

    @application.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: object, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request",
                    "details": details,
                }
            },
        )

    @application.get("/health")
    async def health() -> dict[str, object]:
        return await backend.health()

    @application.post(
        "/indexes/build", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED
    )
    async def build_index(request: IndexBuildRequest) -> JobResponse:
        job = await backend.enqueue_index(request)
        return JobResponse(job_id=job.id, status=job.status)

    @application.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        return await backend.search(request)

    @application.post("/evaluate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
    async def evaluate(request: EvaluateRequest) -> JobResponse:
        job = await backend.enqueue_evaluation(request)
        return JobResponse(job_id=job.id, status=job.status)

    @application.get("/runs/{run_id}", response_model=JobRecord)
    async def get_run(run_id: str, response: Response) -> JobRecord:
        job = await backend.get_job(run_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail={"code": "not_found", "message": "Run not found"}
            )
        response.headers["Cache-Control"] = "no-store"
        return job

    return application


def service_from_environment() -> BenchmarkService:
    import os

    if os.getenv("RB_USE_DATABASE", "false").lower() not in {"1", "true", "yes"}:
        return InMemoryService()
    from retrieval_benchmark.settings import get_settings

    settings = get_settings()
    from retrieval_benchmark.database import PostgresBenchmarkService, create_engine
    from retrieval_benchmark.providers import CachedEmbedder, HashEmbeddingCache, LocalBGEEmbedder

    embedder = CachedEmbedder(
        LocalBGEEmbedder(), HashEmbeddingCache(settings.cache_directory / "embeddings.json")
    )
    return PostgresBenchmarkService(
        create_engine(settings.database_url),
        chunking=settings.default_chunking,
        strategy=settings.default_strategy,
        embedder=embedder,
    )


app = create_app(service_from_environment())
