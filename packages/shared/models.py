"""Pydantic models shared between the API layer and downstream consumers.

Keeping these in `packages/shared` rather than `apps/api` means workers,
tests, and a future TypeScript codegen step all have one canonical
definition.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["explain", "trace", "debug", "refactor"]


class QueryRequest(BaseModel):
    repo: str = Field(..., description="Repo name (e.g. 'fastapi')")
    mode: Mode = "explain"
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
