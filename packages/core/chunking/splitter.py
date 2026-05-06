"""Split oversize chunks into smaller windowed sub-chunks.

Embedding models lose signal when a single chunk covers hundreds of lines:
the resulting vector is an average of too many concepts to match any single
query well. We cap chunk size and split anything larger into overlapping
windows, preserving the parent symbol name for traceability.
"""
from __future__ import annotations

from .base import Chunk

# Target sizes in lines. Picked so that most chunks fit well under
# text-embedding-3-small's 8192-token context with room to spare.
MAX_LINES: int = 80      # split if a chunk exceeds this
WINDOW_LINES: int = 60   # size of each sub-chunk
OVERLAP_LINES: int = 10  # lines shared between adjacent sub-chunks


def split_oversize(chunks: list[Chunk]) -> list[Chunk]:
    """Return a new list with any oversize chunks split into windows.

    Sub-chunks keep the parent's symbol_name so retrieval can still
    identify the containing function/class. Symbol kind becomes 'window'
    because they're no longer atomic semantic units.
    """
    out: list[Chunk] = []
    for chunk in chunks:
        line_count = chunk.end_line - chunk.start_line + 1
        if line_count <= MAX_LINES:
            out.append(chunk)
            continue
        out.extend(_window_split(chunk))
    return out


def _window_split(chunk: Chunk) -> list[Chunk]:
    lines = chunk.content.splitlines(keepends=True)
    pieces: list[Chunk] = []
    step = WINDOW_LINES - OVERLAP_LINES
    assert step > 0, "WINDOW_LINES must exceed OVERLAP_LINES"

    for offset in range(0, len(lines), step):
        window = lines[offset:offset + WINDOW_LINES]
        if not window:
            break
        text = "".join(window).rstrip("\n")
        if not text.strip():
            continue
        pieces.append(Chunk(
            content=text,
            start_line=chunk.start_line + offset,
            end_line=chunk.start_line + offset + len(window) - 1,
            symbol_name=chunk.symbol_name,  # preserve parent name
            symbol_kind="window",
            imports=list(chunk.imports),
        ))
        # Stop if this window already reached the end
        if offset + WINDOW_LINES >= len(lines):
            break
    return pieces
