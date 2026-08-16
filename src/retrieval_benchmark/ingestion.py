from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from retrieval_benchmark.models import Document


class PythonDocsParser:
    def __init__(self, version: str) -> None:
        self.version = version

    def parse_html(self, html: str, relative_path: str) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.find(role="main") or soup.body
        if main is None:
            raise ValueError(f"document {relative_path} has no content container")
        title_node = main.find("h1") or soup.find("title")
        title = title_node.get_text(" ", strip=True) if title_node else Path(relative_path).stem
        module = None
        if relative_path.startswith("library/"):
            module = Path(relative_path).stem.split(".")[0]
        sections: list[dict[str, str]] = []
        current_heading = title
        current_parts: list[str] = []
        for node in main.find_all(["h1", "h2", "h3", "p", "pre", "li"], recursive=True):
            if node.name in {"h1", "h2", "h3"}:
                if current_parts:
                    sections.append({"heading": current_heading, "text": "\n".join(current_parts)})
                current_heading = node.get_text(" ", strip=True)
                current_parts = []
            else:
                value = node.get_text(" ", strip=True)
                if value:
                    current_parts.append(value)
        if current_parts:
            sections.append({"heading": current_heading, "text": "\n".join(current_parts)})
        text = "\n\n".join(section["text"] for section in sections)
        source_uri = (
            f"https://docs.python.org/release/{self.version}/{relative_path.replace(chr(92), '/')}"
        )
        return Document(
            source_uri=source_uri,
            title=title,
            text=text,
            module=module,
            sections=sections,
            metadata={"corpus": "python-docs", "version": self.version, "path": relative_path},
        )

    def parse_directory(self, root: Path) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(root.rglob("*.html")):
            relative = path.relative_to(root).as_posix()
            try:
                document = self.parse_html(
                    path.read_text(encoding="utf-8", errors="replace"), relative
                )
            except ValueError:
                continue
            if document.text.strip():
                documents.append(document)
        return documents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_corpus(url: str, expected_sha256: str, destination: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "docs.python.org":
        raise ValueError("corpus URL must use HTTPS on docs.python.org")
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("a 64-character SHA-256 checksum is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output)
    actual = sha256_file(temporary)
    if actual.lower() != expected_sha256.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError(f"corpus checksum mismatch: expected {expected_sha256}, got {actual}")
    temporary.replace(destination)
    return destination


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if resolved_destination not in target.parents and target != resolved_destination:
                raise ValueError(f"unsafe path in archive: {member.filename}")
        bundle.extractall(destination)  # noqa: S202


def write_documents(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(document.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def load_documents(path: Path) -> list[Document]:
    return [
        Document.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
