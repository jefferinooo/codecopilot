"""LLM re-ranker: take top-N candidates, return top-K with relevance scores.

The hybrid retrieval stage optimizes for RECALL — cast a wide net and
make sure the right answer is somewhere in the top 50. The re-ranker
optimizes for PRECISION — read each candidate and pick the ones that
actually answer the question.

Implementation:
- Build a compact summary of each candidate (path, symbol, first lines
  of content). Full content would blow the prompt budget for 50 chunks.
- Ask Claude Haiku for a JSON array of {chunk_index, relevance, reason}.
- Parse, sort by relevance, return top-K.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from uuid import UUID

from .. import db
from ..llm.client import LLMClient, RERANK_MODEL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankedHit:
    chunk_id: UUID
    relevance: int   # 1-5; 5 = highly relevant
    reason: str


SYSTEM_PROMPT = """You are a code search re-ranker. You will be given a user
question about a codebase and a list of code chunks retrieved as candidate
answers. Score each chunk's relevance on a 1-5 integer scale:

  5 = directly answers the question
  4 = strongly relevant, mentions the right concept and code
  3 = somewhat relevant, related code but not the main answer
  2 = tangentially relevant
  1 = not relevant

Return STRICT JSON: a list of objects with the keys "index" (int, 0-based),
"relevance" (int 1-5), and "reason" (one short sentence).

Include EVERY input chunk in your output, even if relevance is 1. Do not
include any text outside the JSON array."""


def _format_candidate(idx: int, path: str, symbol: str | None, kind: str,
                      start_line: int, end_line: int, content: str) -> str:
    """Compact representation of one chunk for the re-ranker prompt."""
    sym = symbol or "(window)"
    # Trim each chunk to ~30 lines to keep the prompt under budget.
    lines = content.splitlines()
    if len(lines) > 30:
        lines = lines[:28] + ["    ...", lines[-1]]
    body = "\n".join(lines)
    return (
        f"<chunk index=\"{idx}\" path=\"{path}\" "
        f"symbol=\"{sym}\" kind=\"{kind}\" lines=\"{start_line}-{end_line}\">\n"
        f"{body}\n"
        f"</chunk>"
    )


async def rerank(
    question: str,
    candidate_chunk_ids: list[UUID],
    *,
    llm: LLMClient,
    top_k: int = 8,
) -> list[RerankedHit]:
    """Re-rank candidates by relevance to the question."""
    if not candidate_chunk_ids:
        return []

    rows = await db.fetch(
        """
        SELECT c.id, f.path, c.symbol_name, c.symbol_kind,
               c.start_line, c.end_line, c.content
        FROM chunks c JOIN files f ON f.id = c.file_id
        WHERE c.id = ANY($1::uuid[])
        """,
        candidate_chunk_ids,
    )
    # Preserve the input ordering (RRF order matters as a prior)
    by_id = {r["id"]: r for r in rows}
    ordered = [by_id[cid] for cid in candidate_chunk_ids if cid in by_id]

    candidates_text = "\n\n".join(
        _format_candidate(
            i, r["path"], r["symbol_name"], r["symbol_kind"],
            r["start_line"], r["end_line"], r["content"],
        )
        for i, r in enumerate(ordered)
    )

    user_prompt = (
        f"User question: {question}\n\n"
        f"Candidate chunks (rank order from prior retrieval):\n\n"
        f"{candidates_text}"
    )

    response_text = llm.complete(
        model=RERANK_MODEL,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=4096,
    )

    judgments = _parse_json_array(response_text)

    # Map judgments back to chunk_ids and sort
    hits: list[RerankedHit] = []
    for j in judgments:
        idx = j.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(ordered):
            continue
        relevance = j.get("relevance")
        if not isinstance(relevance, int) or not (1 <= relevance <= 5):
            continue
        hits.append(RerankedHit(
            chunk_id=ordered[idx]["id"],
            relevance=relevance,
            reason=str(j.get("reason", "")),
        ))

    hits.sort(key=lambda h: -h.relevance)
    return hits[:top_k]


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from the model's response.

    Models sometimes wrap JSON in ```json ... ``` fences or add stray
    prose. We try strict parsing first, then fall back to extracting the
    first [...] block.
    """
    text = text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [ ... ] block
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("re-ranker: could not parse JSON, returning empty: %r", text[:200])
    return []
