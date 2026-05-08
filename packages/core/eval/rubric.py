"""LLM-as-judge: score answers on correctness, relevance, completeness.

The judge reads the question, the chunks the retriever produced, and
the assistant's final answer. It returns three 1-5 scores in strict
JSON, plus a one-sentence rationale per dimension.

Design notes:
- Correctness is judged ONLY against the provided chunks, not the
  judge model's training data. This keeps scores meaningful when the
  question is about an arbitrary repo.
- Relevance asks whether the answer addresses the actual question, not
  whether it's well-written.
- Completeness flags critical omissions even if everything stated is
  correct.
- Temperature is 0 for reproducibility.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..llm.client import LLMClient, RERANK_MODEL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Judgment:
    correctness: int
    correctness_reason: str
    relevance: int
    relevance_reason: str
    completeness: int
    completeness_reason: str
    judge_model: str

    @property
    def avg(self) -> float:
        return (self.correctness + self.relevance + self.completeness) / 3


@dataclass(frozen=True)
class ChunkSnippet:
    """Minimal chunk info for the judge prompt -- path + content only."""
    path: str
    start_line: int
    end_line: int
    content: str


SYSTEM_PROMPT = """You are evaluating an AI assistant's answer about a
codebase. You will receive:
  - The user's question
  - The code chunks that were retrieved as context
  - The assistant's answer

Score the answer on three integer 1-5 dimensions:

CORRECTNESS: Are the factual claims in the answer supported by the
provided chunks? Score based ONLY on the provided chunks, not on what
you know about the codebase. A claim that is true in reality but cannot
be verified from the chunks is NOT correct for this dimension.
  5 = every claim is verifiable in the chunks
  4 = minor unverifiable details, core claims correct
  3 = mix of verifiable and unverifiable claims
  2 = central claim unverifiable or contradicted by chunks
  1 = the answer contradicts the chunks

RELEVANCE: Does the answer address the user's actual question?
  5 = directly answers what was asked
  4 = answers most of the question
  3 = adjacent but doesn't directly answer
  2 = barely related
  1 = answers a different question

COMPLETENESS: Is critical information present, given the chunks
available? Penalize ONLY for omissions that are visible in the chunks
but missing from the answer. Don't penalize for missing info that
wasn't retrieved.
  5 = nothing important from the chunks was omitted
  4 = minor omissions
  3 = some important details missing
  2 = significant omissions that change meaning
  1 = misleading because of what's missing

Return STRICT JSON with this exact shape:

{
  "correctness":   {"score": <int 1-5>, "reason": "<one sentence>"},
  "relevance":     {"score": <int 1-5>, "reason": "<one sentence>"},
  "completeness":  {"score": <int 1-5>, "reason": "<one sentence>"}
}

Do not include any text outside the JSON object. Do not award full
marks (5) for vague, hedged, or refusal answers; if the assistant
correctly refuses because the chunks don't answer the question, that
gets 4 or 5 on relevance but should be calibrated honestly.

Be a strict but fair judge. The point is to detect quality changes
over time, not to flatter the system."""


def judge(
    question: str,
    chunks: list[ChunkSnippet],
    answer: str,
    *,
    llm: LLMClient,
) -> Judgment | None:
    """Run a single judgment. Returns None if the model output couldn't be parsed."""
    if not answer.strip():
        return None

    chunks_block = "\n".join(_format_chunk(c) for c in chunks)
    user = (
        f"<question>\n{question}\n</question>\n\n"
        f"<chunks>\n{chunks_block}\n</chunks>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )

    response = llm.complete(
        model=RERANK_MODEL,  # Haiku is cheap + good enough for this
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=1024,
        temperature=0.0,
    )

    parsed = _parse_judgment(response)
    if parsed is None:
        return None

    try:
        return Judgment(
            correctness=int(parsed["correctness"]["score"]),
            correctness_reason=str(parsed["correctness"].get("reason", "")),
            relevance=int(parsed["relevance"]["score"]),
            relevance_reason=str(parsed["relevance"].get("reason", "")),
            completeness=int(parsed["completeness"]["score"]),
            completeness_reason=str(parsed["completeness"].get("reason", "")),
            judge_model=RERANK_MODEL,
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("judge: malformed JSON shape: %s", e)
        return None


def _format_chunk(c: ChunkSnippet) -> str:
    return (
        f'<chunk path="{c.path}" lines="{c.start_line}-{c.end_line}">\n'
        f"{c.content}\n"
        f"</chunk>"
    )


def _parse_judgment(text: str) -> dict | None:
    """Extract a JSON object from the judge response, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("judge: could not parse JSON: %r", text[:200])
    return None
