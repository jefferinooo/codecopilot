"""FastAPI application entrypoint.

Routes are registered here; everything else lives in routers/.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from packages.core import db
from .routers.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the DB pool eagerly so the first request doesn't pay the cost
    await db.get_pool()
    yield
    await db.close_pool()


app = FastAPI(
    title="CodeCopilot",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness probe. Returns 200 if the process is up."""
    return {"status": "ok"}


@app.get("/repos")
async def list_repos():
    """List indexed repos and their chunk counts. Useful for the UI later."""
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
