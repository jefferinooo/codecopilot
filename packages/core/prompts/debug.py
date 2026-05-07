"""Debug mode: produce ranked hypotheses for why something might be failing.

Use cases:
  - "Why might this 422 be returned for a valid request?"
  - "Why might dependency injection be returning None?"
  - "What could cause this websocket to disconnect immediately?"

Output is structured: Hypotheses (ranked) -> Evidence (per hypothesis)
-> Next checks. The model is explicitly told NOT to propose fixes --
the goal is diagnosis, not prescription.
"""
from __future__ import annotations

from .explain import ChunkContext


SYSTEM_PROMPT = """You are debugging the issue described by the user.
You will be given a question and code chunks retrieved from a codebase.

Produce three sections, in order:

**Hypotheses**
A ranked list (most likely first) of plausible root causes. Each
hypothesis must be tied to a specific file and line range using the
format `path/to/file.py:42-78`. State each hypothesis as a single
sentence.

**Evidence**
For each hypothesis, in the same order, give the lines or behavior in
the chunks that support OR contradict it. Be honest about contradictions
-- a hypothesis that the evidence contradicts should still appear, with
the contradiction surfaced.

**Next checks**
Concrete diagnostic steps the user could run: log statements to add,
specific values to inspect, commands to execute, branches to step
through. These should be cheap and quick -- not "rewrite this module."

Do NOT propose fixes. Do not write code. The goal is diagnosis only.
A user who knows the root cause will write the fix themselves; a user
who has the wrong root cause will write a fix that doesn't work.

Use ONLY the provided chunks. Do not invent error conditions or
control flow that doesn't appear in the chunks. If the chunks don't
contain enough information to form even one credible hypothesis, say
so plainly in place of the Hypotheses section."""


def build_user_prompt(question: str, chunks: list[ChunkContext]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "(No relevant code was retrieved.)\n\n"
            "Tell the user the codebase doesn't contain enough context "
            "to diagnose this issue, and suggest what kind of code they "
            "would need to share for a real diagnosis."
        )

    max_rel = max(c.relevance for c in chunks)
    weak_note = ""
    if max_rel < 4:
        weak_note = (
            f"\n\nNote: retrieved chunks were judged only weakly relevant "
            f"(max relevance: {max_rel}/5). Hypotheses should be hedged "
            "or you should explicitly say the chunks don't support a "
            "credible diagnosis."
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
