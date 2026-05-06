"""Walk a repository and decide which files to chunk.

Yields (relative_path, language, source_bytes) for files that survive
filtering. Filtering is conservative: when in doubt, skip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Top-level directory names to skip wholesale, anywhere in the tree.
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "dist", "build", "out", ".next", ".turbo", "target",
    ".idea", ".vscode",
    "vendor", "third_party",
})

# File extensions we know how to chunk well.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py":  "python",
    ".pyi": "python",
}

# Filenames or suffixes we never index regardless of extension.
SKIP_FILENAME_SUFFIXES: tuple[str, ...] = (
    ".min.js", ".min.css",
    ".map",
    ".lock",
    "-lock.json",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "uv.lock",
)

# Maximum file size we'll bother chunking (bytes). Larger files are usually
# generated, vendored, or otherwise not useful for retrieval.
MAX_FILE_BYTES: int = 500_000


@dataclass(frozen=True)
class WalkedFile:
    relative_path: Path  # path relative to the repo root
    language: str        # 'python' for now; more later
    source: bytes        # raw file bytes


def is_binary(sample: bytes) -> bool:
    """Heuristic: a NUL byte in the first 8 KB means binary."""
    return b"\x00" in sample[:8192]


def walk_repo(root: Path) -> Iterator[WalkedFile]:
    """Yield every source file under `root` that survives filtering."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    for path in root.rglob("*"):
        # Skip directories themselves; we only care about files
        if not path.is_file():
            continue

        # Any ancestor in SKIP_DIRS? (rglob doesn't filter; we do)
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue

        # Skip lockfiles, minified files, source maps
        name = path.name
        if any(name.endswith(s) for s in SKIP_FILENAME_SUFFIXES):
            continue

        # Language detection by extension
        ext = path.suffix.lower()
        language = LANGUAGE_BY_EXT.get(ext)
        if language is None:
            continue

        # Size guard
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        # Read once, check binary, yield
        try:
            source = path.read_bytes()
        except OSError:
            continue
        if is_binary(source):
            continue

        yield WalkedFile(
            relative_path=path.relative_to(root),
            language=language,
            source=source,
        )
