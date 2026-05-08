"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from packages.core import db
from .routers.eval import router as eval_router
from .routers.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.get_pool()
    yield
    await db.close_pool()


app = FastAPI(title="CodeCopilot", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/repos")
async def list_repos():
    rows = await db.fetch(
        """
        SELECT r.name, r.status, r.ingested_at,
               (SELECT count(*) FROM chunks WHERE repo_id = r.id) AS chunks
        FROM repos r
        ORDER BY r.ingested_at DESC
        """
    )
    return [dict(r) for r in rows]


app.include_router(query_router)
app.include_router(eval_router)
