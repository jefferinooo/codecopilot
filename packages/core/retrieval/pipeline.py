"""Two-stage retrieval: hybrid (RRF) → LLM re-rank."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ..llm.client import LLMClient
from ..llm.embeddings import EmbeddingClient
from .hybrid import hybrid_search
from .reranker import RerankedHit, rerank


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: UUID
    relevance: int
    reason: str


async def retrieve(
    repo_id: UUID,
    question: str,
    *,
    embedder: EmbeddingClient,
    llm: LLMClient,
    candidates_per_strategy: int = 30,
    candidates_for_rerank: int = 50,
    top_k: int = 8,
) -> list[RerankedHit]:
    """Full retrieval pipeline: hybrid search → re-rank → top-K."""
    fused = await hybrid_search(
        repo_id, question, embedder,
        k_per_strategy=candidates_per_strategy,
        k_final=candidates_for_rerank,
    )
    if not fused:
        return []

    return await rerank(
        question,
        [h.chunk_id for h in fused],
        llm=llm,
        top_k=top_k,
    )
