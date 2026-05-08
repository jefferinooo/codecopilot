"""GET /eval/{repo_name}: aggregated eval scores for a repo.

Returns:
- overall: average correctness/relevance/completeness across all judged queries
- by_mode: same, broken down by mode (explain, trace, debug, refactor)
- recent: the 20 most recent judged queries with their scores
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from packages.core import db

router = APIRouter()


@router.get("/eval/{repo_name}")
async def get_eval(repo_name: str):
    repo_id = await db.fetchval("SELECT id FROM repos WHERE name = $1", repo_name)
    if repo_id is None:
        raise HTTPException(status_code=404, detail=f"repo {repo_name!r} not found")

    overall = await db.fetchrow(
        """
        SELECT
          COUNT(*) AS n,
          ROUND(AVG(j.correctness)::numeric, 2)  AS correctness,
          ROUND(AVG(j.relevance)::numeric, 2)    AS relevance,
          ROUND(AVG(j.completeness)::numeric, 2) AS completeness,
          ROUND(AVG((j.correctness + j.relevance + j.completeness) / 3.0)::numeric, 2) AS avg
        FROM judgments j
        JOIN queries q ON q.id = j.query_id
        WHERE q.repo_id = $1
        """,
        repo_id,
    )

    by_mode = await db.fetch(
        """
        SELECT
          q.mode,
          COUNT(*) AS n,
          ROUND(AVG(j.correctness)::numeric, 2)  AS correctness,
          ROUND(AVG(j.relevance)::numeric, 2)    AS relevance,
          ROUND(AVG(j.completeness)::numeric, 2) AS completeness,
          ROUND(AVG((j.correctness + j.relevance + j.completeness) / 3.0)::numeric, 2) AS avg
        FROM judgments j
        JOIN queries q ON q.id = j.query_id
        WHERE q.repo_id = $1
        GROUP BY q.mode
        ORDER BY q.mode
        """,
        repo_id,
    )

    recent = await db.fetch(
        """
        SELECT
          q.id,
          q.mode,
          q.question,
          q.created_at,
          q.latency_ms,
          j.correctness,
          j.relevance,
          j.completeness
        FROM queries q
        LEFT JOIN judgments j ON j.query_id = q.id
        WHERE q.repo_id = $1
        ORDER BY q.created_at DESC
        LIMIT 20
        """,
        repo_id,
    )

    return {
        "repo": repo_name,
        "overall": dict(overall) if overall else None,
        "by_mode": [dict(r) for r in by_mode],
        "recent": [dict(r) for r in recent],
    }
