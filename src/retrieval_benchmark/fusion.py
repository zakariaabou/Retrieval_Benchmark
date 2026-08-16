from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedResult:
    id: str
    score: float
    component_ranks: list[int | None]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], rrf_k: int = 60) -> list[FusedResult]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    all_ids: list[str] = []
    seen: set[str] = set()
    positions: list[dict[str, int]] = []
    for ranking in rankings:
        position = {identifier: rank for rank, identifier in enumerate(ranking, start=1)}
        positions.append(position)
        for identifier in ranking:
            if identifier not in seen:
                seen.add(identifier)
                all_ids.append(identifier)
    fused = [
        FusedResult(
            id=identifier,
            score=sum(
                1.0 / (rrf_k + position[identifier])
                for position in positions
                if identifier in position
            ),
            component_ranks=[position.get(identifier) for position in positions],
        )
        for identifier in all_ids
    ]
    return sorted(fused, key=lambda item: (-item.score, all_ids.index(item.id)))
