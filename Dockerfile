FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra local-models --extra hosted --extra tracking
ENV PATH="/app/.venv/bin:$PATH"
COPY configs ./configs
COPY migrations ./migrations
COPY datasets ./datasets
COPY reports ./reports

FROM runtime AS api
CMD ["uvicorn", "retrieval_benchmark.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS worker
CMD ["python", "-m", "retrieval_benchmark.worker"]

FROM runtime AS mlflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql://benchmark:benchmark@postgres:5432/benchmark", "--default-artifact-root", "/mlartifacts"]
