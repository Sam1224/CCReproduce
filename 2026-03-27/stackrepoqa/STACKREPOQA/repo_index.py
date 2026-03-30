from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dataset import RepoChunk


def iter_repo_files(repo_root: str, exts: Iterable[str]) -> List[Path]:
    root = Path(repo_root)
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if any(part in {".git", "__pycache__", "node_modules", "dist", "build"} for part in p.parts):
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return out


def chunk_file(path: Path, repo_root: Path, max_lines: int = 120, overlap: int = 20) -> List[RepoChunk]:
    rel = str(path.relative_to(repo_root))
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    chunks: List[RepoChunk] = []
    start = 1
    while start <= len(lines):
        end = min(len(lines), start + max_lines - 1)
        text = "\n".join(lines[start - 1 : end])
        chunk_id = f"{rel}:{start}-{end}"
        chunks.append(RepoChunk(chunk_id=chunk_id, path=rel, start_line=start, end_line=end, text=text))
        if end == len(lines):
            break
        start = max(1, end - overlap + 1)

    return chunks


def build_corpus(repo_root: str, exts: Iterable[str] = (".py", ".md")) -> List[RepoChunk]:
    root = Path(repo_root).resolve()
    files = iter_repo_files(str(root), exts=set(exts))

    corpus: List[RepoChunk] = []
    for p in files:
        corpus.extend(chunk_file(p, root))

    # Make chunk_ids stable.
    corpus.sort(key=lambda c: c.chunk_id)
    return corpus
