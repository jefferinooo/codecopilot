"""Refactor mode: propose a minimum-viable change as a unified diff."""
from __future__ import annotations

from .explain import ChunkContext


SYSTEM_PROMPT = """You are proposing a refactor to a codebase. You will
be given a question describing the desired change and code chunks
retrieved from the codebase.

Produce three sections, in order:

**Diff**
The smallest change that addresses the user's goal, formatted as a
unified diff against the cited file. Use exact paths from the <chunk>
tags. Include 2-3 lines of context above and below the change. Do not
include unchanged regions of the file.

**Rationale**
Up to 4 short bullet points explaining the reasoning. Each bullet
should reference specific lines (`path/to/file.py:42-78`) where
possible.

**Risks**
A list of concrete risks introduced by this change. Cover at minimum:
- Tests that would likely need updating.
- Public APIs or behaviors that change observably.
- Performance implications, if relevant.
- Edge cases the new code handles differently from the old code.

If the user's goal is unsafe, ambiguous, or the chunks don't contain
enough context to propose a confident change, refuse and explain why
in place of the Diff section. A refused refactor with a clear
explanation is more useful than a confident-looking diff that breaks
the system.

Prefer the minimum viable change over the most elegant change. A 5-line
diff that solves the problem beats a 200-line rewrite. Use ONLY the
provided chunks; do not invent functions, classes, or imports that
don't appear in the chunks."""


def build_user_prompt(question: str, chunks: list[ChunkContext]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "(No relevant code was retrieved.)\n\n"
            "Tell the user the codebase doesn't appear to contain the "
            "code described in the request, so a confident refactor "
            "isn't possible without seeing the actual code."
        )

    max_rel = max(c.relevance for c in chunks)
    weak_note = ""
    if max_rel < 4:
        weak_note = (
            f"\n\nNote: retrieved chunks were judged only weakly relevant "
            f"(max relevance: {max_rel}/5). If the chunks don't actually "
            "contain the code the user wants refactored, refuse the "
            "request rather than refactoring tangentially-related code."
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
