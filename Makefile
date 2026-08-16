.PHONY: install test coverage lint typecheck up down ingest

install:
	uv sync --frozen --extra dev

test:
	python -m pytest -q

coverage:
	python -m pytest --cov=retrieval_benchmark --cov-report=term-missing

lint:
	python -m ruff check .

typecheck:
	python -m mypy src

up:
	docker compose up --build -d

down:
	docker compose down

ingest:
	retrieval-benchmark corpus download
	retrieval-benchmark corpus ingest
