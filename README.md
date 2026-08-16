# Production RAG Retrieval Benchmark

A reproducible benchmark for comparing PostgreSQL lexical retrieval, exact and HNSW dense
retrieval, Reciprocal Rank Fusion, and cross-encoder reranking under quality, latency, and
cost constraints. Retrieval is the product; answer generation is intentionally out of scope.

## What is implemented

- Five deterministic fixed/structural chunking configurations with content-derived IDs.
- PostgreSQL English full-text search and pgvector exact/HNSW cosine retrieval.
- Local BGE and OpenAI embedding adapters with a content-addressed cache.
- RRF hybrid retrieval plus local MiniLM and Cohere reranking adapters.
- Recall/precision/nDCG/context metrics at 1, 5, 10, and 20, plus MRR and latency percentiles.
- Human-review, dataset validation, test-freezing, and checksum enforcement workflows.
- MLflow run metadata/artifact logging and static Markdown/HTML reports.
- FastAPI search and durable PostgreSQL-backed build/evaluation jobs claimed with
  `FOR UPDATE SKIP LOCKED`.
- Docker Compose services for PostgreSQL/pgvector, MLflow, API, and worker.

## Quick start

Requirements: Python 3.11 and Docker Desktop. Copy `.env.example` to `.env`. Hosted
credentials are optional unless you run the OpenAI/Cohere configurations.

```bash
uv sync --frozen --extra dev
python -m pytest
docker compose up --build -d
curl http://localhost:8000/health
```

Before downloading the corpus, replace the checksum in `datasets/corpus-manifest.json` with
the independently verified SHA-256 for the official Python 3.14 archive. The downloader
deliberately refuses an absent or malformed checksum.

```bash
retrieval-benchmark corpus download
retrieval-benchmark corpus ingest
```

The resulting archive and parsed corpus are ignored by Git. The committed manifest supplies
the corpus version, source URL, license, and checksum.

## Evaluation data

Evaluation results are meaningful only after human verification. Prepare 300 records in
`datasets/evaluation/all.jsonl`, balanced as 100 development and 200 test queries across the
five query categories and three difficulty levels. The included example demonstrates the
schema but is intentionally marked `draft`.

```bash
retrieval-benchmark dataset seed
retrieval-benchmark dataset validate datasets/evaluation/all.jsonl
retrieval-benchmark dataset review datasets/evaluation/all.jsonl
retrieval-benchmark dataset freeze
retrieval-benchmark dataset validate datasets/evaluation/test.jsonl \
  --verified --manifest datasets/evaluation/test.manifest.json
```

Do all parameter selection on `dev`. A configuration with `split: test` and `tuning: true`
is rejected. Do not edit `test.jsonl` after freezing; evaluation verifies its digest.

## Build indexes and run searches

With the stack running, enqueue an index build:

```bash
curl -X POST http://localhost:8000/indexes/build \
  -H "Content-Type: application/json" \
  -d '{"corpus_version":"3.14.6","chunking":"fixed_256_o32","embedding_provider":"local","distance":"cosine"}'
```

Poll the returned ID through `GET /runs/{run_id}`. Search uses the local hybrid champion by
default:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I create a virtual environment?","top_k":10,"strategy":"hybrid","filters":{"module":"venv"}}'
```

Enqueue evaluation with a configuration path visible inside the worker container:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"configuration":"configs/experiments/04_local_hybrid.yaml","split":"test"}'
```

MLflow is available at <http://localhost:5000> and the API at <http://localhost:8000/docs>.

## Experiment protocol

The nine frozen configurations in `configs/experiments/` cover lexical, local/hosted exact
and HNSW dense retrieval, local/hosted hybrid retrieval, and local/hosted reranking. Tune on
development data over:

- RRF constant: 20, 60, 100.
- Candidate depth: 50, 100, 200.
- Rerank candidates: 20, 50, 100.
- HNSW `m`: 8, 16, 32; `ef_construction`: 64, 128; `ef_search`: 20, 40, 80, 160.

Run the frozen test only after selecting settings. Record warm-up policy, five shuffled local
passes, machine profile, index time/size, peak RSS, API usage, and the dated pricing snapshot.
Exact pgvector search is produced transactionally by disabling index and bitmap scans.

Load-test the selected local champion after 50 warm-up requests:

```bash
python scripts/load_test.py --concurrency 1 --duration 60
python scripts/load_test.py --concurrency 4 --duration 60
python scripts/load_test.py --concurrency 8 --duration 60
```

## API contracts

- `POST /indexes/build` — enqueue corpus chunking, embedding, and indexing.
- `POST /search` — search by explicit strategy and safe allowlisted metadata filters.
- `POST /evaluate` — enqueue an immutable experiment configuration.
- `GET /runs/{run_id}` — return job state, progress, result, or structured error.
- `GET /health` — report API, PostgreSQL, and pgvector readiness.

The local deployment is intentionally unauthenticated. Do not expose it to an untrusted
network. Secrets come from environment variables and are never logged as run parameters.

## Development checks

```bash
make test
make coverage
make lint
make typecheck
```

CI never calls paid providers. Provider contracts are tested with fakes, while paid final
runs are manual and credential-gated.

## Required final analysis

The generated report supplies a global table and three recommendation slots. A completed
benchmark must additionally populate category/difficulty slices, quality/latency Pareto
plots, HNSW recall curves, chunking and fusion/reranking ablations, and twenty representative
errors. Do not claim benchmark numbers until all 300 records have been human-reviewed.

## CV bullets after a real run

> Benchmarked lexical, dense and hybrid retrieval on **[N] verified queries**; hybrid
> retrieval with reranking improved Recall@5 from **[baseline]** to **[result]**.

> Evaluated HNSW search and reranking under load, achieving **[latency] ms p95** at **[QPS]**
> while preserving **[recall]**.

> Packaged the selected retrieval pipeline as a tested FastAPI service and tracked datasets,
> parameters and results with MLflow.
