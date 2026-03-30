from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import List, Tuple

from dataset import QAExample, RepoQADataset, save_dataset
from repo_index import build_corpus


def extract_python_qas(repo_root: Path, max_examples: int = 200) -> List[Tuple[str, str, str]]:
    # (question, answer, support_chunk_id)
    out: List[Tuple[str, str, str]] = []

    py_files = [p for p in repo_root.rglob("*.py") if p.is_file() and ".git" not in p.parts]
    py_files.sort()

    for path in py_files:
        if len(out) >= max_examples:
            break

        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        rel = str(path.relative_to(repo_root))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = getattr(node, "name", "")
            doc = ast.get_docstring(node) or ""
            if not name or not doc.strip():
                continue
            question = f"What does {name} do in {rel}?"
            answer = doc.strip().splitlines()[0][:200]

            # Support: use the chunk containing the definition line.
            lineno = getattr(node, "lineno", 1)
            start = max(1, lineno - 20)
            end = lineno + 80
            support_chunk_id = f"{rel}:{start}-{end}"
            out.append((question, answer, support_chunk_id))

            if len(out) >= max_examples:
                break

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-examples", type=int, default=200)
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    corpus = build_corpus(str(repo_root), exts=(".py", ".md"))

    triples = extract_python_qas(repo_root, max_examples=args.max_examples)

    # Map support chunk id to real corpus id: use best-effort matching by prefix.
    corpus_ids = {c.chunk_id: c.chunk_id for c in corpus}

    examples = []
    for i, (q, a, sid) in enumerate(triples):
        if sid not in corpus_ids:
            # fall back to any chunk in the same file.
            file_prefix = sid.split(":")[0] + ":"
            cand = [c.chunk_id for c in corpus if c.chunk_id.startswith(file_prefix)]
            if not cand:
                continue
            sid = cand[0]

        examples.append(QAExample(qid=f"q{i}", question=q, answer=a, support_chunk_id=sid))

    ds = RepoQADataset(chunks=corpus, examples=examples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(ds, str(out_path))
    print(f"Wrote dataset: {out_path} (chunks={len(ds.chunks)} examples={len(ds.examples)})")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
