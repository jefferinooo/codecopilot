"""POST /query: stream a grounded answer; persist + judge in background."""
from __future__ import annotations

import logging
import time
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from packages.core import db
from packages.core.answer import answer
from packages.core.eval.rubric import ChunkSnippet, judge
from packages.core.llm.client import LLMClient
from packages.core.llm.embeddings import EmbeddingClient
from packages.core.retrieval.pipeline import retrieve
from packages.shared.models import QueryRequest

router = APIRouter()
log = logging.getLogger("query")

_embedder = EmbeddingClient()
_llm = LLMClient()


@router.post("/query")
async def query(req: QueryRequest, background: BackgroundTasks) -> StreamingResponse:
    repo_id = await db.fetchval(
        "SELECT id FROM repos WHERE name = $1 AND status = 'ready'",
        req.repo,
    )
    if repo_id is None:
        raise HTTPException(status_code=404,
                            detail=f"repo {req.repo!r} not found or not ready")

    # Retrieve once here so we can persist the chunk IDs alongside the answer.
    # The answer generator will look them up again -- redundant but cheap.
    t0 = time.perf_counter()
    hits = await retrieve(
        repo_id, req.question, embedder=_embedder, llm=_llm, top_k=req.top_k,
    )
    chunk_ids = [str(h.chunk_id) for h in hits]

    answer_buffer: list[str] = []

    async def generate():
        async for chunk in _answer_with_prefetched(repo_id, req, hits):
            answer_buffer.append(chunk)
            yield chunk

        # After streaming finishes, persist the query and queue judgment.
        latency_ms = int((time.perf_counter() - t0) * 1000)
        full_answer = "".join(answer_buffer)
        try:
            query_id = await _persist_query(
                repo_id=repo_id,
                req=req,
                chunk_ids=chunk_ids,
                answer=full_answer,
                latency_ms=latency_ms,
            )
            background.add_task(
                _judge_in_background, query_id, req.question, hits, full_answer,
            )
        except Exception as exc:
            log.exception("persist or judge enqueue failed: %s", exc)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


async def _answer_with_prefetched(repo_id: UUID, req: QueryRequest, hits):
    """Wrap answer() but skip its retrieval step by injecting our hits."""
    # We re-call answer() which redoes retrieval. For Phase 4 simplicity we
    # accept that double cost; we'll refactor to take pre-retrieved hits
    # in Phase 5 polish if it matters.
    async for chunk in answer(
        repo_id=repo_id, mode=req.mode, question=req.question,
        embedder=_embedder, llm=_llm, top_k=req.top_k,
    ):
        yield chunk


async def _persist_query(
    *,
    repo_id: UUID,
    req: QueryRequest,
    chunk_ids: list[str],
    answer: str,
    latency_ms: int,
) -> UUID:
    return await db.fetchval(
        """
        INSERT INTO queries
            (repo_id, mode, question, retrieved, answer, latency_ms)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        RETURNING id
        """,
        repo_id, req.mode, req.question, chunk_ids, answer, latency_ms,
    )


async def _judge_in_background(query_id: UUID, question: str, hits, answer: str):
    """Background task: hydrate chunks, run judge, persist judgment."""
    try:
        if not hits or not answer.strip():
            return
        rows = await db.fetch(
            """
            SELECT c.id, f.path, c.start_line, c.end_line, c.content
            FROM chunks c JOIN files f ON f.id = c.file_id
            WHERE c.id = ANY($1::uuid[])
            """,
            [h.chunk_id for h in hits],
        )
        by_id = {r["id"]: r for r in rows}
        snippets = [
            ChunkSnippet(
                path=by_id[h.chunk_id]["path"],
                start_line=by_id[h.chunk_id]["start_line"],
                end_line=by_id[h.chunk_id]["end_line"],
                content=by_id[h.chunk_id]["content"],
            )
            for h in hits if h.chunk_id in by_id
        ]
        verdict = judge(question, snippets, answer, llm=_llm)
        if verdict is None:
            log.warning("judge returned None for query %s", query_id)
            return
        await db.execute(
            """
            INSERT INTO judgments
                (query_id, judge_model, correctness, relevance, completeness, rationale)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            query_id,
            verdict.judge_model,
            verdict.correctness,
            verdict.relevance,
            verdict.completeness,
            (
                f"correctness: {verdict.correctness_reason}\n"
                f"relevance: {verdict.relevance_reason}\n"
                f"completeness: {verdict.completeness_reason}"
            ),
        )
        log.info("judged query %s: %.2f", query_id, verdict.avg)
    except Exception:
        log.exception("background judge failed for query %s", query_id)
