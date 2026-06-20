"""Anchor-aware text generation (SAMA Sec III.D-4).

Nucleus (top-p) sampling that varies syntax while keeping the semantic anchors
(entity / trigger / argument words) invariant -> the projected labels stay valid.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from anchors import anchor_ids, entity_anchor_words
from data import SAMPLES, STOI, VOCAB, Sample, encode

# content tokens that may be resampled (the surface words seen in the corpus,
# excluding the pinned anchor words, which are decided per-sample)
CONTENT_WORDS = sorted({w for s in SAMPLES for w in s.tokens})
CONTENT_IDS = torch.tensor([STOI[w] for w in CONTENT_WORDS])


@torch.no_grad()
def generate_text(model, sample: Sample, p=0.9) -> List[int]:
    ids = torch.tensor([encode(sample.tokens)])
    aids = torch.tensor([anchor_ids(sample)])
    logits, _, _, _ = model(ids, sample.image.unsqueeze(0), aids, sample.task)
    pinned = set(entity_anchor_words(sample))
    out = [ids[0, 0].item()]
    for t in range(1, ids.size(1)):
        word = sample.tokens[t]
        if word in pinned:
            out.append(ids[0, t].item())                # keep anchor token
            continue
        # restrict to content vocab, apply top-p nucleus sampling on model logits
        lg = logits[0, t - 1, CONTENT_IDS]
        probs = F.softmax(lg, -1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, 0) <= p
        keep[0] = True
        idx = si[keep]
        pr = probs[idx] / probs[idx].sum()
        choice = idx[torch.multinomial(pr, 1)].item()
        out.append(int(CONTENT_IDS[choice].item()))
    return out


def decode(ids: List[int]) -> str:
    return " ".join(VOCAB[i] for i in ids)
