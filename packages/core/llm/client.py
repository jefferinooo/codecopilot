"""Anthropic Claude client.

Two entrypoints:
- complete(): one-shot completion. Used for structured-output tasks
  (re-ranking, judging) where we want the full response at once.
- stream(): yields text chunks as they arrive. Used for answer
  generation where users see tokens appear in real time.

Both retry on transient errors with exponential backoff. Non-retryable
errors (auth, validation) raise immediately.
"""
from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator, Iterable

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError

logger = logging.getLogger(__name__)

# Model choices, picked for their roles:
# - Haiku is cheap + fast, ideal for the re-ranker (structured judgment over
#   ~50 short summaries).
# - Sonnet is the workhorse for answer generation: better reasoning, still
#   cost-effective, supports streaming well.
RERANK_MODEL: str = "claude-haiku-4-5-20251001"
ANSWER_MODEL: str = "claude-sonnet-4-6"

MAX_RETRIES: int = 5
INITIAL_BACKOFF_SECONDS: float = 1.0


class LLMClient:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env "
                "(loaded by direnv) and try again."
            )
        self._client = Anthropic(api_key=key)

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """One-shot completion. Returns the model's text response."""
        return self._with_retry(
            lambda: self._client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        ).content[0].text

    def stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Iterable[str]:
        """Streamed completion. Yields text chunks as the model produces them.

        This is the synchronous streaming API. We'll wrap it in async later
        when we wire it into FastAPI's StreamingResponse.
        """
        with self._client.messages.stream(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ---------- internal ----------

    def _with_retry(self, call):
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return call()
            except (RateLimitError, APIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    logger.error("llm: giving up after %d attempts: %s", attempt, e)
                    raise
                logger.warning(
                    "llm: transient error (attempt %d/%d): %s; sleeping %.1fs",
                    attempt, MAX_RETRIES, type(e).__name__, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            except APIError as e:
                logger.error("llm: non-retryable API error: %s", e)
                raise
        raise RuntimeError("unreachable")
