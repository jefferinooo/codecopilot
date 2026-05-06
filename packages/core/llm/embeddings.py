"""Embedding client wrapping OpenAI's text-embedding-3-small.

Design notes:
- Batches into groups of 96 because that's the sweet spot for throughput
  vs. memory on text-embedding-3-small. Each batch is one HTTP request.
- Retries on transient errors (rate limits, network blips) with exponential
  backoff. Does NOT retry on authentication or validation errors.
- Returns vectors as plain Python lists; the DB layer converts to whatever
  pgvector expects.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

from openai import OpenAI, APIConnectionError, APIError, RateLimitError

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "text-embedding-3-small"
EMBED_DIM: int = 1536            # must match the VECTOR(1536) column
BATCH_SIZE: int = 96             # OpenAI sweet spot
MAX_RETRIES: int = 5
INITIAL_BACKOFF_SECONDS: float = 1.0


class EmbeddingClient:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env "
                "(loaded by direnv) and try again."
            )
        self._client = OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, in the same order."""
        if not texts:
            return []
        out: list[list[float]] = []
        for batch in _chunks(texts, BATCH_SIZE):
            out.extend(self._embed_batch_with_retry(batch))
        return out

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                return [item.embedding for item in response.data]
            except (RateLimitError, APIConnectionError) as e:
                if attempt == MAX_RETRIES:
                    logger.error("embed: giving up after %d attempts: %s", attempt, e)
                    raise
                logger.warning(
                    "embed: transient error (attempt %d/%d): %s; sleeping %.1fs",
                    attempt, MAX_RETRIES, type(e).__name__, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            except APIError as e:
                # Non-transient API error (e.g. invalid request). Don't retry.
                logger.error("embed: non-retryable API error: %s", e)
                raise
        raise RuntimeError("unreachable")  # for type checker


def _chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
