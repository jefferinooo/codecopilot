"""Keyword (full-text) search over chunk content using Postgres tsvector.

Uses ts_rank_cd, which is Postgres's BM25-ish ranking function. Higher
scores mean better matches. Conceptually:
  - Tokens are extracted from the user's question
  - Each chunk's content_tsv is scored against those tokens
  - Top-K chunks by score are returned
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .. import db


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: UUID
    score: float


async def keyword_search(
    repo_id: UUID,
    question: str,
    k: int = 30,
) -> list[KeywordHit]:
    """Return the top-K chunks for `repo_id` matching `question`.

    Uses websearch_to_tsquery: handles natural-language queries gracefully
    (quoted phrases, OR, negation) without requiring users to write tsquery
    syntax. ts_rank_cd accounts for both term frequency and document length.
    """
    rows = await db.fetch(
        """
        SELECT
            c.id,
            ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', $2)) AS score
        FROM chunks c
        WHERE c.repo_id = $1
          AND c.content_tsv @@ websearch_to_tsquery('english', $2)
        ORDER BY score DESC
        LIMIT $3
        """,
        repo_id, question, k,
    )
    return [KeywordHit(chunk_id=row["id"], score=row["score"]) for row in rows]
