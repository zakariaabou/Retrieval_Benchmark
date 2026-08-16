# Security review

Reviewed boundaries: FastAPI input, PostgreSQL queries, evaluation configuration paths,
corpus download/extraction, worker errors, external provider secrets, generated HTML, and
Docker exposure.

## Resolved findings

- All request bodies use bounded Pydantic models; evaluation paths must be relative YAML files
  under `configs/`, and the worker independently verifies the resolved path.
- SQL values are bound parameters. The only dynamic identifiers are embedding columns and
  metadata filter columns selected from fixed internal allowlists.
- ZIP members are resolved and checked against the extraction root before extraction.
- Corpus downloads require HTTPS, the exact `docs.python.org` host, and a pinned SHA-256.
- Worker exceptions are logged server-side while API-visible jobs receive a generic error.
- Concurrent searches create a request-scoped retriever instead of mutating shared strategy.
- Compose binds PostgreSQL, MLflow, and FastAPI to `127.0.0.1`; the local unauthenticated stack
  is not exposed on the LAN.
- API responses set `nosniff`, frame denial, and no-referrer headers. `.env` and caches are
  ignored, and run metadata excludes credentials.

## Accepted local-only constraints

Authentication, CSRF, distributed rate limiting, and TLS termination are intentionally absent
for v1. They are mandatory before changing the deployment from loopback-only to a shared or
public environment.

