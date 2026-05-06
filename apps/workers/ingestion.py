"""Ingest a local repository: walk, chunk, split, report stats.

Usage:
    python -m apps.workers.ingestion /path/to/repo

In Phase 1d this will also embed chunks and persist them to Postgres.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from packages.core.chunking.python_chunker import PythonChunker
from packages.core.chunking.splitter import split_oversize
from packages.core.chunking.walker import walk_repo

CHUNKERS = {
    "python": PythonChunker(),
}


def ingest(repo_path: Path, *, verbose: bool = False) -> None:
    files_seen = 0
    files_chunked = 0
    total_chunks = 0
    total_lines = 0
    kinds: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    biggest: list[tuple[int, str, str]] = []

    for walked in walk_repo(repo_path):
        files_seen += 1
        chunker = CHUNKERS.get(walked.language)
        if chunker is None:
            continue

        chunks = chunker.chunk(walked.source)
        chunks = split_oversize(chunks)        # NEW
        files_chunked += 1
        total_chunks += len(chunks)
        total_lines += walked.source.count(b"\n") + 1
        languages[walked.language] += 1

        for c in chunks:
            kinds[c.symbol_kind] += 1
            size = c.end_line - c.start_line + 1
            biggest.append((size, str(walked.relative_path), c.symbol_name or "(window)"))

        if verbose:
            print(f"{walked.relative_path}  ->  {len(chunks)} chunks")

    biggest.sort(reverse=True)

    print()
    print("=" * 70)
    print(f"Ingestion summary: {repo_path}")
    print("=" * 70)
    print(f"Files seen:    {files_seen}")
    print(f"Files chunked: {files_chunked}")
    print(f"Lines:         {total_lines}")
    print(f"Chunks:        {total_chunks}")
    if files_chunked:
        print(f"Avg chunks/file: {total_chunks / files_chunked:.1f}")
    print()
    print("By language:")
    for lang, n in languages.most_common():
        print(f"  {lang:<12} {n:>5}")
    print()
    print("By chunk kind:")
    for kind, n in kinds.most_common():
        print(f"  {kind:<10} {n:>5}")
    print()
    print("10 largest chunks (lines):")
    for size, path, sym in biggest[:10]:
        print(f"  {size:>4}  {path:<40}  {sym}")


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest a local repository.")
    p.add_argument("path", type=Path, help="path to a local repository")
    p.add_argument("-v", "--verbose", action="store_true", help="print per-file output")
    args = p.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 1

    ingest(args.path, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
