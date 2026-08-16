from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


def _winner(rows: list[dict[str, Any]], key: str, maximize: bool) -> str:
    if not rows:
        return "No completed run"
    eligible = [row for row in rows if key in row]
    if not eligible:
        return "Not available"
    selected = (max if maximize else min)(eligible, key=lambda row: float(row[key]))
    return str(selected.get("name", "unnamed"))


def write_report(rows: list[dict[str, Any]], output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if table.empty:
        markdown_table = "_No completed runs._"
    else:
        columns = [str(column) for column in table.columns]
        markdown_lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        markdown_lines.extend(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in table.itertuples(index=False, name=None)
        )
        markdown_table = "\n".join(markdown_lines)
    quality = _winner(rows, "recall@5", True)
    latency = _winner(rows, "p95_ms", False)
    cost = _winner(rows, "cost_usd", False)
    markdown = f"""# Retrieval Benchmark Report

## Global comparison

{markdown_table}

## Recommendations

- **Maximum quality:** {quality}
- **Minimum latency:** {latency}
- **Minimum cost:** {cost}

## Required analyses

Category and difficulty breakdowns, chunking/HNSW tradeoffs, fusion and reranker ablations,
and twenty representative errors are populated by the final benchmark runner artifacts.
"""
    markdown_path = output / "benchmark.md"
    html_path = output / "benchmark.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    comparison = table.to_html(index=False) if not table.empty else "<p>No completed runs.</p>"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Retrieval Benchmark</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.45rem}</style></head>"
        f"<body><h1>Retrieval Benchmark Report</h1>{comparison}"
        f"<h2>Recommendations</h2><ul><li>Maximum quality: {html.escape(quality)}</li>"
        f"<li>Minimum latency: {html.escape(latency)}</li>"
        f"<li>Minimum cost: {html.escape(cost)}</li></ul></body></html>",
        encoding="utf-8",
    )
    return markdown_path, html_path
