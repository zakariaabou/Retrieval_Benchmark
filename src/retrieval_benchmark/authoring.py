from __future__ import annotations

from typing import Literal

from retrieval_benchmark.models import BenchmarkQuery, Document, PassageJudgment

Category = Literal["factual", "paraphrase", "exact_keyword", "acronym", "multi_passage"]
CATEGORIES: tuple[Category, ...] = (
    "factual",
    "paraphrase",
    "exact_keyword",
    "acronym",
    "multi_passage",
)


def _question(category: Category, document: Document, second: Document) -> str:
    topic = document.title.rstrip("¶")
    module = document.module or topic.split()[0]
    if category == "factual":
        return f"What Python capability is described in {topic}?"
    if category == "paraphrase":
        return f"How does the Python documentation explain {topic}?"
    if category == "exact_keyword":
        return f"Where is the exact Python term '{module}' documented?"
    if category == "acronym":
        return f"What does the documentation say about the acronym or identifier {module.upper()}?"
    return f"How do {topic} and {second.title.rstrip('¶')} relate in Python?"


def build_draft_queries(
    documents: list[Document], total: int = 300, dev_count: int = 100
) -> list[BenchmarkQuery]:
    if len(documents) < 2:
        raise ValueError("at least two documents are required")
    if total <= 0 or dev_count < 0 or dev_count >= total:
        raise ValueError("invalid total/dev_count")
    if total % len(CATEGORIES) or dev_count % len(CATEGORIES):
        raise ValueError("total and dev_count must be divisible by five categories")
    per_category = total // len(CATEGORIES)
    dev_per_category = dev_count // len(CATEGORIES)
    output: list[BenchmarkQuery] = []
    for category_index, category in enumerate(CATEGORIES):
        for ordinal in range(per_category):
            document_index = (category_index * per_category + ordinal) % len(documents)
            document = documents[document_index]
            second = documents[(document_index + 1) % len(documents)]
            passages = [
                PassageJudgment(
                    document_id=document.id,
                    start=0,
                    end=max(1, min(len(document.text.split()), 500)),
                    grade=2,
                )
            ]
            if category == "multi_passage":
                passages.append(
                    PassageJudgment(
                        document_id=second.id,
                        start=0,
                        end=max(1, min(len(second.text.split()), 500)),
                        grade=1,
                    )
                )
            output.append(
                BenchmarkQuery(
                    id=f"py314-{category}-{ordinal + 1:03d}",
                    question=_question(category, document, second),
                    category=category,
                    difficulty=("easy", "medium", "hard")[ordinal % 3],
                    split="dev" if ordinal < dev_per_category else "test",
                    relevant_passages=passages,
                    reference_answer=document.text[:500].strip(),
                    provenance=f"Python 3.14.6: {document.source_uri}",
                    generation_method="deterministic-template-seed",
                    reviewer=None,
                    validated_at=None,
                    status="draft",
                )
            )
    return output
