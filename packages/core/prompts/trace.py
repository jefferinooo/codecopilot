"""Trace mode: walk through how something flows across the codebase.

Use cases:
  - "Where does request authentication get checked?"
  - "Trace what happens when a websocket message arrives."
  - "How does a path parameter get from the URL to the handler argument?"

The output is a numbered list of steps. Each step names the file +
function and quotes the line that performs the step.
"""
from __future__ import annotations

from .explain import ChunkContext


SYSTEM_PROMPT = """You are tracing a request, data flow, or control flow
through a codebase. You will be given a question and code chunks
retrieved from the codebase.

Output a numbered list of steps. Each step MUST:

1. Name the file and the function or class involved.
2. Cite the exact line range as `path/to/file.py:42-78` (use the line
   ranges from the <chunk> tags).
3. Briefly describe what that step does and what it passes to the
   next step.

After the numbered steps, include a "Gaps" section listing any steps
in the flow that you could not verify from the provided chunks. If
the chunks don't trace a coherent path through the codebase, say so
directly in the Gaps section -- do not fabricate steps to fill in
missing pieces.

Use ONLY the provided chunks. Do not invent function names, parameters,
or behaviors that don't appear in the chunks."""


def build_user_prompt(question: str, chunks: list[ChunkContext]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "(No relevant code was retrieved.)\n\n"
            "Tell the user the codebase doesn't appear to contain code "
            "that traces this flow."
        )

    max_rel = max(c.relevance for c in chunks)
    weak_note = ""
    if max_rel < 4:
        weak_note = (
            f"\n\nNote: retrieved chunks were judged only weakly relevant "
            f"(max relevance: {max_rel}/5). The flow may be partially "
            "or entirely missing from the provided context. Surface this "
            "honestly in the Gaps section."
        )

    chunks_xml = "\n".join(_format_chunk(c) for c in chunks)
    return (
        f"Question: {question}\n\n"
        f"<chunks>\n{chunks_xml}\n</chunks>"
        f"{weak_note}"
    )


def _format_chunk(c: ChunkContext) -> str:
    sym_attr = f' symbol="{c.symbol_name}"' if c.symbol_name else ""
    return (
        f'<chunk path="{c.path}"{sym_attr} kind="{c.symbol_kind}" '
        f'lines="{c.start_line}-{c.end_line}" relevance="{c.relevance}">\n'
        f"{c.content}\n"
        f"</chunk>"
    )
