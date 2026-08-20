from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _dcg(relevances: Sequence[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(relevances)))


def evaluate_ranking(
    retrieved: Sequence[str], judgments: Mapping[str, int], ks: Iterable[int] = (1, 5, 10, 20)
) -> dict[str, float]:
    # A chunk can only be relevant once. De-duplicate defensively so malformed
    # provider output cannot inflate recall above 1.0 or distort rank metrics.
    ranking = list(dict.fromkeys(retrieved))
    relevant = {identifier for identifier, grade in judgments.items() if grade > 0}
    result: dict[str, float] = {}
    for k in ks:
        top = ranking[:k]
        hits = sum(identifier in relevant for identifier in top)
        result[f"recall@{k}"] = hits / len(relevant) if relevant else 0.0
        result[f"precision@{k}"] = hits / k
        grades = [judgments.get(identifier, 0) for identifier in top]
        ideal = sorted(judgments.values(), reverse=True)[:k]
        ideal_dcg = _dcg(ideal)
        result[f"ndcg@{k}"] = _dcg(grades) / ideal_dcg if ideal_dcg else 0.0
        result[f"context_precision@{k}"] = hits / k
        result[f"context_recall@{k}"] = hits / len(relevant) if relevant else 0.0
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, identifier in enumerate(ranking, start=1)
            if identifier in relevant
        ),
        0.0,
    )
    result["mrr"] = reciprocal_rank
    return result


def aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}
