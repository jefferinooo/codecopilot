"""POST /query: stream a grounded answer for a question about a repo.

Pipeline (all hidden behind one HTTP request):
  request → look up repo_id by name → retrieve+rerank → stream answer
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from packages.core import db
from packages.core.answer import answer
from packages.core.llm.client import LLMClient
from packages.core.llm.embeddings import EmbeddingClient
from packages.shared.models import QueryRequest

router = APIRouter()

# These are stateless and safe to instantiate once per process.
# A real production setup would inject them via FastAPI's Depends(),
# but for a portfolio project module-level singletons are clean enough.
_embedder = EmbeddingClient()
_llm = LLMClient()


@router.post("/query")
async def query(req: QueryRequest) -> StreamingResponse:
    repo_id = await db.fetchval(
        "SELECT id FROM repos WHERE name = $1 AND status = 'ready'",
        req.repo,
    )
    if repo_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"repo {req.repo!r} not found or not ready",
        )

    async def generate():
        async for chunk in answer(
            repo_id=repo_id,
            mode=req.mode,
            question=req.question,
            embedder=_embedder,
            llm=_llm,
            top_k=req.top_k,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
