"""
evaluate.py — Offline metrics for OneBar (paper Sec. 5.1 "Evaluation Metrics").

Given a (trained or random) generator, decode K=8 candidate queries per
trigger and compute:

  * Exact HR@8   : hit if the ground-truth query exactly matches one of top-8.
  * MRR@8        : mean reciprocal rank of the first exact hit.
  * ED-HR@8      : hit if Levenshtein(pred, gt) <= 2 for some top-8 candidate.
  * BLEU@8       : best sentence-level BLEU among top-8 (sacrebleu).
  * BAS@8        : Behavior-Aligned Similarity — PLACEHOLDER. The paper uses a
                   collaborative-data-trained embedding model; we approximate
                   it with a lexical char-n-gram cosine and clearly flag it.

Example (smoke test):
    python evaluate.py --tiny --n_eval 12
    python evaluate.py --init_from outputs/piopd --n_eval 24
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import List

import torch
from torch.utils.data import DataLoader

from data import generate_dataset, build_prompt
from model import OneBarGenerator

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def sentence_bleu(pred: str, ref: str) -> float:
    if _HAS_SACREBLEU:
        return sacrebleu.sentence_bleu(pred, [ref]).score / 100.0
    # fallback: char-bigram F1 as a rough BLEU proxy
    return _char_ngram_cosine(pred, ref)


def _char_ngram_counter(s: str, n: int = 2) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(max(len(s) - n + 1, 0))) or Counter(s)


def _char_ngram_cosine(a: str, b: str, n: int = 2) -> float:
    """Lexical char-n-gram cosine — used for the BAS@8 PLACEHOLDER.

    Real BAS@8 (paper): similarity from a collaborative-data-trained embedding
    model between trigger evidence and each generated query, averaged over top-K.
    Replace `_char_ngram_cosine` with that embedding cosine for a faithful BAS.
    """
    ca, cb = _char_ngram_counter(a, n), _char_ngram_counter(b, n)
    common = set(ca) & set(cb)
    dot = sum(ca[g] * cb[g] for g in common)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: OneBarGenerator, samples, K=8, prompt_style="compressed",
             batch_size=4, max_len=24):
    tok = model.tokenizer
    model.eval()

    eval_samples = [s for s in samples if not s.is_reject]
    prompts = [build_prompt(s, prompt_style) for s in eval_samples]
    gts = [s.clicked_query for s in eval_samples]

    hr, mrr, edhr, bleu, bas = 0.0, 0.0, 0.0, 0.0, 0.0
    N = len(eval_samples)

    for i in range(0, N, batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, max_length=128, truncation=True, padding=True,
                  return_tensors="pt")
        preds_batch = model.generate_queries(
            enc["input_ids"], enc["attention_mask"], K=K, max_len=max_len)
        for j, preds in enumerate(preds_batch):
            gt = gts[i + j]
            preds = preds[:K]
            # Exact HR@8 + MRR@8
            hit_rank = None
            for rank, p in enumerate(preds, 1):
                if p == gt:
                    hit_rank = rank
                    break
            if hit_rank is not None:
                hr += 1.0
                mrr += 1.0 / hit_rank
            # ED-HR@8
            if any(levenshtein(p, gt) <= 2 for p in preds):
                edhr += 1.0
            # BLEU@8 (best of top-K)
            if preds:
                bleu += max(sentence_bleu(p, gt) for p in preds)
                bas += max(_char_ngram_cosine(p, gt) for p in preds)

    return {
        "N": N,
        "Exact HR@8": hr / N,
        "MRR@8": mrr / N,
        "ED-HR@8": edhr / N,
        "BLEU@8": bleu / N,
        "BAS@8 (placeholder)": bas / N,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/bart-base")
    ap.add_argument("--init_from", default="")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n_eval", type=int, default=24)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--prompt_style", default="compressed")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    name = args.init_from if args.init_from else args.model_name
    model = OneBarGenerator(name, tiny=args.tiny, offline=args.offline, K=args.K)
    samples = generate_dataset(n=args.n_eval, seed=args.seed)
    metrics = evaluate(model, samples, K=args.K, prompt_style=args.prompt_style)
    print("\n=== OneBar offline evaluation (toy set) ===")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    if not _HAS_SACREBLEU:
        print("  [note] sacrebleu not installed -> BLEU uses char-ngram proxy.")


if __name__ == "__main__":
    main()
