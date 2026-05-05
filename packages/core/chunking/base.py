"""Shared types and base class for all language chunkers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Chunk:
    """One unit of retrieval. Maps 1:1 to a row in the chunks table."""

    content: str
    start_line: int          # 1-indexed, inclusive
    end_line: int            # 1-indexed, inclusive
    symbol_name: str | None  # e.g. "Greeter.greet" or "ingest_repo"
    symbol_kind: str         # 'function' | 'class' | 'method' | 'window'
    imports: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.start_line > self.end_line:
            raise ValueError(
                f"start_line {self.start_line} > end_line {self.end_line} "
                f"for {self.symbol_name!r}"
            )
        if self.symbol_kind not in {"function", "class", "method", "window"}:
            raise ValueError(f"invalid symbol_kind: {self.symbol_kind!r}")


class Chunker(Protocol):
    """Every language chunker implements this interface."""

    def chunk(self, source: bytes) -> list[Chunk]: ...
