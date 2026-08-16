from pathlib import Path

from retrieval_benchmark.reporting import write_report


def test_report_contains_required_recommendations(tmp_path: Path) -> None:
    markdown, html = write_report(
        [{"name": "lexical", "recall@5": 0.5, "p95_ms": 5.0, "cost_usd": 0.0}],
        tmp_path,
    )
    text = markdown.read_text(encoding="utf-8")
    assert "Maximum quality" in text
    assert "Minimum latency" in text
    assert "Minimum cost" in text
    assert html.exists()
