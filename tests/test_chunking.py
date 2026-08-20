from retrieval_benchmark.chunking import Chunker, stable_id
from retrieval_benchmark.models import ChunkingConfig, Document


def test_fixed_chunking_is_deterministic_and_overlaps() -> None:
    doc = Document(
        source_uri="https://example.test/doc",
        title="Doc",
        text=" ".join(f"w{i}" for i in range(20)),
    )
    config = ChunkingConfig(
        name="tiny", mode="fixed", target_tokens=8, max_tokens=8, overlap_tokens=2
    )

    first = Chunker().chunk(doc, config)
    second = Chunker().chunk(doc, config)

    assert first == second
    assert first[0].token_end == 8
    assert first[1].token_start == 6
    assert first[0].id == stable_id(doc.id, config.name, "0", first[0].text)


def test_structural_chunks_preserve_heading_metadata() -> None:
    doc = Document(
        source_uri="https://example.test/doc",
        title="Doc",
        text="Install Python first. Then create a virtual environment.",
        module="venv",
        sections=[
            {"heading": "Installation", "text": "Install Python first."},
            {"heading": "Usage", "text": "Then create a virtual environment."},
        ],
    )
    config = ChunkingConfig(
        name="structural", mode="structural", target_tokens=5, max_tokens=8, overlap_tokens=1
    )

    chunks = Chunker().chunk(doc, config)

    assert {chunk.heading for chunk in chunks} == {"Installation", "Usage"}
    assert all(chunk.document_id == doc.id for chunk in chunks)
    assert all(chunk.metadata["module"] == "venv" for chunk in chunks)
