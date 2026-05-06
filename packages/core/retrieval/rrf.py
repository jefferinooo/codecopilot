"""Reciprocal Rank Fusion: combine multiple ranked lists into one.

RRF is the standard robust way to fuse rankings with different score
scales. It's nearly parameter-free (a single `k` constant), it ignores
raw score magnitudes, and empirically beats most learned alternatives
at this scale.

Reference: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class FusedHit:
    chunk_id: UUID
    score: float
    sources: list[str]  # which rankings contributed; useful for debugging


def reciprocal_rank_fusion(
    rankings: dict[str, list[UUID]],
    k: int = 60,
) -> list[FusedHit]:
    """Fuse named ranked lists into one ranking.

    Each input is a list of chunk_ids in best-first order. The output
    is sorted by fused score (descending). A chunk's score is the sum
    over all rankings where it appears, of 1/(k + rank).

    `k` dampens the influence of any single ranking. The standard
    choice (60) means rank 1 is worth 1/61 ≈ 0.016, rank 100 is worth
    1/160 ≈ 0.006. A chunk ranked first in two lists outranks one
    ranked first in only one list.
    """
    scores: dict[UUID, float] = {}
    sources: dict[UUID, list[str]] = {}
    for source, ranking in rankings.items():
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(chunk_id, []).append(source)

    return sorted(
        (FusedHit(chunk_id=cid, score=s, sources=sources[cid])
         for cid, s in scores.items()),
        key=lambda h: -h.score,
    )
