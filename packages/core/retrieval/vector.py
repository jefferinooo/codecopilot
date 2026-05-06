"""Vector (semantic) search using pgvector cosine distance."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .. import db


@dataclass(frozen=True)
class VectorHit:
    chunk_id: UUID
    similarity: float


async def vector_search(
    repo_id: UUID,
    query_embedding: list[float],
    k: int = 30,
) -> list[VectorHit]:
    """Return top-K chunks most similar to query_embedding."""
    rows = await db.fetch(
        """
        SELECT id, 1 - (embedding <=> $1) AS similarity
        FROM chunks
        WHERE repo_id = $2
        ORDER BY embedding <=> $1
        LIMIT $3
        """,
        query_embedding, repo_id, k,
    )
    return [VectorHit(chunk_id=r["id"], similarity=r["similarity"]) for r in rows]
