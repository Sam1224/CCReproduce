from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch

from dataset import load_dataset
from model import TextEncoder, make_batch


def normalize_tokens(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]


def f1(pred: str, gold: str) -> float:
    p = normalize_tokens(pred)
    g = normalize_tokens(gold)
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(set(p))
    rec = len(common) / len(set(g))
    return 2 * prec * rec / (prec + rec)


def extractive_answer(chunk_text: str) -> str:
    # Toy answerer: try to extract the first docstring line (this matches how the
    # toy dataset is built), otherwise fall back to the first non-empty line.
    lines = chunk_text.splitlines()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            rest = line[3:]

            # One-liner docstring: """foo"""
            if rest.endswith(quote):
                inner = rest[: -len(quote)].strip()
                if inner:
                    return inner[:200]

            # Multi-line docstring: grab first non-empty content line.
            for j in range(i + 1, len(lines)):
                inner = lines[j].strip()
                if inner.endswith(quote):
                    inner = inner[: -len(quote)].strip()
                if not inner:
                    continue
                return inner[:200]
            break

    for raw in lines:
        line = raw.strip()
        if line:
            return line[:200]

    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="checkpoints/retriever.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset(args.data)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = TextEncoder(in_dim=int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    chunks = ds.chunks
    chunk_texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    # Pre-encode corpus.
    with torch.no_grad():
        # Encode in small batches.
        d_embs = []
        for i in range(0, len(chunk_texts), 64):
            b = make_batch(["q"] * min(64, len(chunk_texts) - i), chunk_texts[i : i + 64], in_dim=int(ckpt["in_dim"]))
            d = b.d_pos.to(device)
            d_embs.append(model(d))
        D = torch.cat(d_embs, dim=0)

    chunk_by_id = {c.chunk_id: c.text for c in chunks}

    em = 0
    f1_sum = 0.0
    n = 0
    hit = 0

    for ex in ds.examples[:200]:
        with torch.no_grad():
            bq = make_batch([ex.question], ["dummy"], in_dim=int(ckpt["in_dim"]))
            q = bq.q.to(device)
            q_emb = model(q)
            sims = (q_emb @ D.t()).squeeze(0)
            best = int(torch.argmax(sims).item())

        best_id = chunk_ids[best]
        if best_id == ex.support_chunk_id:
            hit += 1

        # Oracle-extractive answering: use the annotated support chunk text.
        # This keeps the toy evaluation meaningful even when the retriever is weak.
        oracle_text = chunk_by_id.get(ex.support_chunk_id, chunk_texts[best])
        pred = extractive_answer(oracle_text)
        gold = ex.answer

        if pred.strip().lower() == gold.strip().lower():
            em += 1
        f1_sum += f1(pred, gold)
        n += 1

    print(
        f"examples={n} retrieval@1={hit/max(1,n):.3f} "
        f"oracle_EM={em/max(1,n):.3f} oracle_F1={f1_sum/max(1,n):.3f}"
    )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
