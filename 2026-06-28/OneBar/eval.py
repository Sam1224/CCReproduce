from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from data import Session, encode
from model import pad_2d


def decode(ids: List[int], inv_vocab: Dict[int, str], eos_id: int, pad_id: int) -> List[str]:
    out = []
    for t in ids:
        if t == eos_id:
            break
        if t == pad_id:
            continue
        tok = inv_vocab.get(int(t), "")
        if tok in {"<bos>", "<eos>", "<pad>"}:
            continue
        out.append(tok)
    return out


def token_f1(pred: List[str], gold: List[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    p = {}
    g = {}
    for t in pred:
        p[t] = p.get(t, 0) + 1
    for t in gold:
        g[t] = g.get(t, 0) + 1
    inter = 0
    for t, c in p.items():
        inter += min(c, g.get(t, 0))
    prec = inter / max(1, len(pred))
    rec = inter / max(1, len(gold))
    if prec + rec <= 1e-9:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def evaluate(model, samples: Sequence[Session], vocab: Dict[str, int], device: str = "cpu") -> Dict[str, float]:
    inv = {i: t for t, i in vocab.items()}
    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    src_ids = [encode(s.evidence_tokens, vocab) for s in samples]
    src = pad_2d(src_ids, pad_id=pad_id).to(device)
    src_pad = src.eq(pad_id)

    gen = model.greedy_decode(src, src_pad, bos_id=bos_id, eos_id=eos_id)

    em = []
    f1 = []
    for i, s in enumerate(samples):
        pred_toks = decode(gen[i].tolist(), inv, eos_id=eos_id, pad_id=pad_id)
        gold_toks = list(s.target_tokens)
        em.append(float(pred_toks == gold_toks))
        f1.append(token_f1(pred_toks, gold_toks))

    return {"exact_match": float(np.mean(em)), "token_f1": float(np.mean(f1))}
