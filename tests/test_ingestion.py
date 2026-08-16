import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from retrieval_benchmark.ingestion import (
    PythonDocsParser,
    download_corpus,
    extract_zip,
    load_documents,
    write_documents,
)


def test_parser_extracts_title_sections_code_and_canonical_url() -> None:
    html = """
    <html><head><title>venv — Creation</title></head><body>
      <main><section id="venv"><h1>venv — Creation</h1>
      <p>The venv module supports creating environments.</p>
      <pre>python -m venv .venv</pre><h2>API</h2><p>Use EnvBuilder.</p>
      </section></main>
    </body></html>
    """
    doc = PythonDocsParser("3.14.6").parse_html(html, "library/venv.html")
    assert doc.title == "venv — Creation"
    assert doc.module == "venv"
    assert doc.source_uri.endswith("library/venv.html")
    assert "python -m venv" in doc.text
    assert any(section["heading"] == "API" for section in doc.sections)


def test_directory_round_trip_and_verified_download(tmp_path: Path, monkeypatch) -> None:
    html_root = tmp_path / "html"
    html_root.mkdir()
    (html_root / "index.html").write_text(
        "<html><body><main><h1>Index</h1><p>Useful content.</p></main></body></html>",
        encoding="utf-8",
    )
    documents = PythonDocsParser("3.14.6").parse_directory(html_root)
    output = tmp_path / "documents.jsonl"
    write_documents(documents, output)
    assert load_documents(output) == documents

    payload = b"official archive"

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(payload))
    archive = download_corpus(
        "https://docs.python.org/archive.zip",
        hashlib.sha256(payload).hexdigest(),
        tmp_path / "a.zip",
    )
    assert archive.read_bytes() == payload
    with pytest.raises(ValueError, match="docs.python.org"):
        download_corpus("https://example.test/a.zip", "0" * 64, tmp_path / "bad.zip")


def test_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe path"):
        extract_zip(archive, tmp_path / "out")
