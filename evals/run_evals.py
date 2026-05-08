"""Run the golden set against the live system, aggregate scores.

Usage:
    python evals/run_evals.py --repo fastapi
    python evals/run_evals.py --repo fastapi --tag baseline-2026-05-07
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from packages.core import db
from packages.core.answer import answer
from packages.core.eval.rubric import ChunkSnippet, judge
from packages.core.llm.client import LLMClient
from packages.core.llm.embeddings import EmbeddingClient
from packages.core.retrieval.pipeline import retrieve


async def run_one(repo_id, item, embedder, llm):
    """Run a single golden question and return its judgment."""
    question = item["question"]
    mode = item["mode"]

    hits = await retrieve(repo_id, question, embedder=embedder, llm=llm, top_k=8)

    rows = await db.fetch(
        """
        SELECT c.id, f.path, c.start_line, c.end_line, c.content
        FROM chunks c JOIN files f ON f.id = c.file_id
        WHERE c.id = ANY($1::uuid[])
        """,
        [h.chunk_id for h in hits],
    )
    by_id = {r["id"]: r for r in rows}
    snippets = [
        ChunkSnippet(
            path=by_id[h.chunk_id]["path"],
            start_line=by_id[h.chunk_id]["start_line"],
            end_line=by_id[h.chunk_id]["end_line"],
            content=by_id[h.chunk_id]["content"],
        )
        for h in hits if h.chunk_id in by_id
    ]
    retrieved_paths = sorted({s.path for s in snippets})

    answer_text = ""
    async for piece in answer(
        repo_id=repo_id, mode=mode, question=question,
        embedder=embedder, llm=llm,
    ):
        answer_text += piece

    verdict = judge(question, snippets, answer_text, llm=llm)
    return {
        "id": item["id"],
        "mode": mode,
        "question": question,
        "retrieved_paths": retrieved_paths,
        "answer_chars": len(answer_text),
        "judgment": (
            None if verdict is None
            else {
                "correctness": verdict.correctness,
                "relevance": verdict.relevance,
                "completeness": verdict.completeness,
                "avg": round(verdict.avg, 2),
            }
        ),
    }


async def main(repo_name, tag):
    repo_id = await db.fetchval(
        "SELECT id FROM repos WHERE name = $1 AND status = 'ready'", repo_name,
    )
    if repo_id is None:
        print(f"error: repo {repo_name!r} not found or not ready", file=sys.stderr)
        await db.close_pool()
        sys.exit(1)

    golden_file = _REPO_ROOT / "evals" / "golden" / f"repo_{repo_name}.jsonl"
    if not golden_file.exists():
        print(f"error: no golden set at {golden_file}", file=sys.stderr)
        await db.close_pool()
        sys.exit(1)

    items = [json.loads(line) for line in golden_file.read_text().splitlines() if line.strip()]
    print(f"running {len(items)} questions against {repo_name}...")

    embedder = EmbeddingClient()
    llm = LLMClient()

    results = []
    t0 = time.perf_counter()
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['id']}: {item['question'][:60]}...", flush=True)
        try:
            r = await run_one(repo_id, item, embedder, llm)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            r = {"id": item["id"], "mode": item["mode"], "question": item["question"],
                 "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if r.get("judgment"):
            j = r["judgment"]
            print(f"    -> c={j['correctness']} r={j['relevance']} co={j['completeness']} avg={j['avg']}")

    elapsed = time.perf_counter() - t0

    judged = [r for r in results if r.get("judgment")]
    if not judged:
        print("\nno successful judgments; aborting aggregation")
        await db.close_pool()
        return

    overall = {
        "correctness":  round(mean(r["judgment"]["correctness"]  for r in judged), 2),
        "relevance":    round(mean(r["judgment"]["relevance"]    for r in judged), 2),
        "completeness": round(mean(r["judgment"]["completeness"] for r in judged), 2),
    }
    overall["avg"] = round(mean(overall.values()), 2)

    by_mode = defaultdict(lambda: {"n": 0, "c": [], "r": [], "co": []})
    for r in judged:
        b = by_mode[r["mode"]]
        b["n"] += 1
        b["c"].append(r["judgment"]["correctness"])
        b["r"].append(r["judgment"]["relevance"])
        b["co"].append(r["judgment"]["completeness"])
    by_mode_summary = {
        m: {
            "n": v["n"],
            "correctness":  round(mean(v["c"]),  2),
            "relevance":    round(mean(v["r"]),  2),
            "completeness": round(mean(v["co"]), 2),
        }
        for m, v in by_mode.items()
    }

    print()
    print("=" * 70)
    print(f"GOLDEN SET RESULTS  ({repo_name}, {len(judged)}/{len(items)} judged, {elapsed:.0f}s)")
    if tag:
        print(f"tag: {tag}")
    print("=" * 70)
    print(f"correctness:  {overall['correctness']}/5")
    print(f"relevance:    {overall['relevance']}/5")
    print(f"completeness: {overall['completeness']}/5")
    print(f"AVERAGE:      {overall['avg']}/5")
    print()
    print("by mode:")
    for m, s in sorted(by_mode_summary.items()):
        print(f"  {m:<10} n={s['n']}  c={s['correctness']}  r={s['relevance']}  co={s['completeness']}")

    out_dir = _REPO_ROOT / "evals" / "results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{repo_name}_{tag}_{timestamp}.json" if tag else f"{repo_name}_{timestamp}.json"
    out_path = out_dir / name
    out_path.write_text(json.dumps({
        "repo": repo_name,
        "tag": tag,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 1),
        "overall": overall,
        "by_mode": by_mode_summary,
        "results": results,
    }, indent=2))
    print(f"\nresults saved to {out_path.relative_to(_REPO_ROOT)}")
    await db.close_pool()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run golden-set evaluation.")
    p.add_argument("--repo", required=True, help="repo name (e.g. fastapi)")
    p.add_argument("--tag", help="optional label for this run")
    args = p.parse_args()
    asyncio.run(main(args.repo, args.tag))
