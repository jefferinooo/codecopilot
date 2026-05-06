"""Async Postgres client with pgvector support.

Design notes:
- Uses asyncpg's connection pool. Each query borrows a connection,
  runs, returns it. No per-query TCP overhead.
- Registers a vector codec on every new connection so we can pass
  Python lists directly into VECTOR columns and read them back.
- Single global pool, lazily initialized.
"""
from __future__ import annotations

import json
import os
from typing import Any

import asyncpg


_pool: asyncpg.Pool | None = None


def _normalize_url(url: str) -> str:
    """asyncpg wants 'postgresql://' or 'postgres://', not SQLAlchemy's
    'postgresql+psycopg://' driver-prefixed form."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://"):]
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://"):]
    return url


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Set up codecs and session settings for every new pool connection."""
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(str(x) for x in v) + "]",
        decoder=lambda s: [float(x) for x in s.strip("[]").split(",")] if s else [],
        schema="public",
        format="text",
    )
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )
    # IVFFlat default probes=1 returns empty results under selective WHERE
    # filters because only one cluster is searched. 10 is the standard
    # recommendation for lists=100. We'll migrate to HNSW later.
    await conn.execute("SET ivfflat.probes = 10")


async def get_pool() -> asyncpg.Pool:
    """Return the lazily-initialized global connection pool."""
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. Make sure direnv loaded .env."
            )
        _pool = await asyncpg.create_pool(
            _normalize_url(url),
            min_size=2,
            max_size=10,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    """Close the pool. Call this on shutdown to flush connections cleanly."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def fetchval(query: str, *args: Any) -> Any:
    pool = await get_pool()
    return await pool.fetchval(query, *args)


async def fetch(query: str, *args: Any) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> asyncpg.Record | None:
    pool = await get_pool()
    return await pool.fetchrow(query, *args)


async def execute(query: str, *args: Any) -> str:
    pool = await get_pool()
    return await pool.execute(query, *args)


async def executemany(query: str, args: list[tuple]) -> None:
    pool = await get_pool()
    await pool.executemany(query, args)
