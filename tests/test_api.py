from fastapi.testclient import TestClient

from retrieval_benchmark.api import create_app, service_from_environment
from retrieval_benchmark.services import InMemoryService


def test_health_and_job_endpoints() -> None:
    client = TestClient(create_app(InMemoryService()))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post(
        "/indexes/build",
        json={
            "corpus_version": "3.14.6",
            "chunking": "fixed_256_o32",
            "embedding_provider": "local",
            "distance": "cosine",
        },
    )
    assert created.status_code == 202
    run = client.get(f"/runs/{created.json()['job_id']}")
    assert run.status_code == 200
    assert run.json()["status"] == "queued"


def test_search_validates_query_and_returns_provenance() -> None:
    client = TestClient(create_app(InMemoryService()))
    assert client.post("/search", json={"query": "", "top_k": 5}).status_code == 422
    response = client.post("/search", json={"query": "virtual environments", "top_k": 5})
    assert response.status_code == 200
    assert "elapsed_ms" in response.json()
    assert response.json()["results"] == []


def test_missing_run_and_evaluation_job() -> None:
    client = TestClient(create_app(InMemoryService()))
    assert client.get("/runs/unknown").status_code == 404
    response = client.post(
        "/evaluate", json={"configuration": "configs/dev_tuning.yaml", "split": "dev"}
    )
    assert response.status_code == 202
    assert client.post("/evaluate", json={"configuration": "../secrets.yaml"}).status_code == 422


def test_service_selection_uses_parsed_settings(monkeypatch) -> None:
    import retrieval_benchmark.settings as settings_module

    class SettingsStub:
        use_database = False

    monkeypatch.setenv("RB_USE_DATABASE", "true")
    monkeypatch.setattr(settings_module, "get_settings", lambda: SettingsStub())

    assert isinstance(service_from_environment(), InMemoryService)
