"""Generate streamed natural-language answers from retrieved chunks."""
from __future__ import annotations

from typing import AsyncIterator
from uuid import UUID

from . import db
from .llm.client import ANSWER_MODEL, LLMClient
from .llm.embeddings import EmbeddingClient
from .prompts.debug import (
    SYSTEM_PROMPT as DEBUG_SYSTEM,
    build_user_prompt as build_debug_user,
)
from .prompts.explain import (
    ChunkContext,
    SYSTEM_PROMPT as EXPLAIN_SYSTEM,
    build_user_prompt as build_explain_user,
)
from .prompts.refactor import (
    SYSTEM_PROMPT as REFACTOR_SYSTEM,
    build_user_prompt as build_refactor_user,
)
from .prompts.trace import (
    SYSTEM_PROMPT as TRACE_SYSTEM,
    build_user_prompt as build_trace_user,
)
from .retrieval.pipeline import retrieve


PROMPTS = {
    "explain":  (EXPLAIN_SYSTEM,  build_explain_user),
    "trace":    (TRACE_SYSTEM,    build_trace_user),
    "debug":    (DEBUG_SYSTEM,    build_debug_user),
    "refactor": (REFACTOR_SYSTEM, build_refactor_user),
}


async def _hydrate_chunks(hits) -> list[ChunkContext]:
    if not hits:
        return []
    rows = await db.fetch(
        """
        SELECT c.id, f.path, c.symbol_name, c.symbol_kind,
               c.start_line, c.end_line, c.content
        FROM chunks c JOIN files f ON f.id = c.file_id
        WHERE c.id = ANY($1::uuid[])
        """,
        [h.chunk_id for h in hits],
    )
    by_id = {r["id"]: r for r in rows}
    out: list[ChunkContext] = []
    for h in hits:
        r = by_id.get(h.chunk_id)
        if r is None:
            continue
        out.append(ChunkContext(
            path=r["path"],
            symbol_name=r["symbol_name"],
            symbol_kind=r["symbol_kind"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            content=r["content"],
            relevance=h.relevance,
        ))
    return out


async def answer(
    *,
    repo_id: UUID,
    mode: str,
    question: str,
    embedder: EmbeddingClient,
    llm: LLMClient,
    top_k: int = 8,
) -> AsyncIterator[str]:
    if mode not in PROMPTS:
        raise ValueError(f"unknown mode: {mode!r}; known: {list(PROMPTS)}")

    system_prompt, build_user = PROMPTS[mode]

    hits = await retrieve(
        repo_id, question,
        embedder=embedder, llm=llm, top_k=top_k,
    )
    contexts = await _hydrate_chunks(hits)
    user_prompt = build_user(question, contexts)

    for piece in llm.stream(
        model=ANSWER_MODEL,
        system=system_prompt,
        user=user_prompt,
    ):
        yield piece
