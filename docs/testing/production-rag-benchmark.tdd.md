# TDD evidence: production retrieval benchmark

## Source and user journeys

The source was the user-approved implementation plan. It was normalized into these journeys:

- A researcher can ingest a checksum-pinned corpus and produce deterministic chunks.
- A reviewer can validate, review, and freeze an immutable evaluation set.
- An engineer can compare lexical, dense, hybrid, and reranked rankings with known metrics.
- An API client can enqueue durable work, search safely, inspect runs, and check health.
- An operator can reproduce dependencies, Docker services, MLflow artifacts, and reports.

## RED/GREEN report

| Guarantee | RED evidence | GREEN evidence |
|---|---|---|
| Core package contracts exist | Initial `python -m pytest -q` failed collection with eight `ModuleNotFoundError` errors | `24 passed` after implementation |
| Ingestion, SQL validation, costing, and reports exist | Focused test run failed collection for four missing modules | Included in the 24-test green run |
| Balanced authoring queue exists | `tests/test_authoring.py` failed with missing `retrieval_benchmark.authoring` | Generated and validated 300 draft records |
| Coverage stays above the agreed gate | First coverage run reported 51.79% | Final branch coverage: 86.29% |

## Final verification

| What is guaranteed | Test or command | Type | Result |
|---|---|---|---|
| Metrics match known rankings and empty cases | `tests/test_metrics.py` | unit | PASS |
| Chunk IDs, overlap, and headings are deterministic | `tests/test_chunking.py` | unit | PASS |
| RRF ordering and reranked hybrid behavior are stable | `tests/test_fusion.py`, `tests/test_retrieval.py` | unit | PASS |
| Test freezing detects mutation and unverified data | `tests/test_dataset.py` | unit | PASS |
| API validates requests, blocks path traversal, and returns jobs | `tests/test_api.py` | integration-style | PASS |
| Corpus download verifies host/checksum and ZIP extraction blocks traversal | `tests/test_ingestion.py` | security/unit | PASS |
| Provider cache avoids duplicate calls | `tests/test_providers.py` | contract | PASS |
| Core branch coverage exceeds 80% | `python -m pytest --cov=retrieval_benchmark --cov-report=term-missing` | coverage | PASS, 86.29% |
| Source passes lint and strict typing | `python -m ruff check .`; `python -m mypy src` | static | PASS |
| Compose schema is valid | `docker compose config --quiet` | configuration | PASS |
| Real corpus parses | `python -m retrieval_benchmark.cli corpus ingest` | smoke | PASS, 569 documents |
| Draft authoring queue validates | `dataset seed`; `dataset validate` | smoke | PASS, 300 drafts |

## Known gaps

- Infrastructure, paid-provider, MLflow, CLI, and worker adapters are excluded from unit
  coverage and are exercised by Docker/manual integration runs. Docker Desktop was not
  running in this environment, so containers and the PostgreSQL migration were configuration-
  validated but not started locally.
- OpenAI and Cohere calls were not made; CI is deliberately credential-free and cost-free.
- The 300 generated records are drafts. A human must inspect every question and source span
  before `dataset freeze`; no automated process is allowed to claim human verification.
- FastAPI's installed TestClient emits one upstream deprecation warning; it does not affect
  test results.
