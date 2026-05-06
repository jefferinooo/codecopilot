"""Hybrid retrieval: vector + keyword + RRF.

This is the production retrieval entrypoint. It runs vector and keyword
searches in parallel, then fuses the rankings with RRF. The fused list
is what we return for re-ranking.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from .. import db
from ..llm.embeddings import EmbeddingClient
from .keyword import keyword_search
from .rrf import FusedHit, reciprocal_rank_fusion
from .vector import vector_search


async def hybrid_search(
    repo_id: UUID,
    question: str,
    embedder: EmbeddingClient,
    k_per_strategy: int = 30,
    k_final: int = 50,
) -> list[FusedHit]:
    """Run vector + keyword in parallel, fuse with RRF, return top-K_final."""
    [query_vec] = embedder.embed([question])

    vector_task = vector_search(repo_id, query_vec, k=k_per_strategy)
    keyword_task = keyword_search(repo_id, question, k=k_per_strategy)
    vec_hits, kw_hits = await asyncio.gather(vector_task, keyword_task)

    fused = reciprocal_rank_fusion({
        "vector":  [h.chunk_id for h in vec_hits],
        "keyword": [h.chunk_id for h in kw_hits],
    })
    return fused[:k_final]
