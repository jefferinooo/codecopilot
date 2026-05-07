"""Explain mode: answer 'what does this code do' questions.

The system prompt establishes the role and constraints. The user prompt
packages the question, the retrieved chunks (with relevance scores), and
a reminder to cite. Citations use {path}:{start_line}-{end_line} so the
UI can later make them clickable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkContext:
    path: str
    symbol_name: str | None
    symbol_kind: str
    start_line: int
    end_line: int
    content: str
    relevance: int  # 1-5 from re-ranker


SYSTEM_PROMPT = """You are a senior engineer reviewing code with a teammate.

You will be given a question and a set of code chunks retrieved from a
codebase. Use ONLY those chunks as your source of truth. Do not invent
function names, file paths, or behaviors that aren't in the chunks.

Format your answer like this:

1. Lead with a one-paragraph high-level explanation in your own words.
2. Then go deeper, citing specific chunks. When citing, use the format
   `path/to/file.py:42-78` (the exact path and line range from the
   <chunk> tags). Inline these citations as part of the prose, e.g.:
   "The dependency tree is built in `dependencies/utils.py:189-260`."
3. If the chunks contradict each other, surface the contradiction
   instead of choosing one side silently.

If the chunks don't actually answer the question, say so directly. Do
not fabricate a confident answer. The user benefits more from "the
codebase doesn't appear to handle this" than from a guess.

Be concrete. Say what the code does, not just what it's about. Use
short paragraphs. No headers, no bullet lists for the answer body
unless the answer is genuinely a list."""


def build_user_prompt(
    question: str,
    chunks: list[ChunkContext],
) -> str:
    """Assemble the user-side prompt with question + cited chunks."""
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "(No relevant code was retrieved.)\n\n"
            "Tell the user the codebase doesn't appear to contain code "
            "that answers this question."
        )

    # If every chunk is weakly relevant, flag it for the model
    max_rel = max(c.relevance for c in chunks)
    weak_context_note = ""
    if max_rel < 4:
        weak_context_note = (
            "\n\nNote: the retrieved chunks were judged only weakly "
            "relevant (max relevance score: "
            f"{max_rel}/5). Consider whether the codebase truly answers "
            "the question, and hedge or decline accordingly."
        )

    chunks_xml = "\n".join(
        _format_chunk(c) for c in chunks
    )

    return (
        f"Question: {question}\n\n"
        f"<chunks>\n{chunks_xml}\n</chunks>"
        f"{weak_context_note}"
    )


def _format_chunk(c: ChunkContext) -> str:
    sym_attr = f' symbol="{c.symbol_name}"' if c.symbol_name else ""
    return (
        f'<chunk path="{c.path}"{sym_attr} kind="{c.symbol_kind}" '
        f'lines="{c.start_line}-{c.end_line}" relevance="{c.relevance}">\n'
        f"{c.content}\n"
        f"</chunk>"
    )
