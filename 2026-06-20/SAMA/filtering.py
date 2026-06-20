"""Dual-Constraint Filtering (SAMA Sec III.F).

Selects synthetic samples by a confidence score balancing cross-modal alignment
and anchor fidelity (Eq. 12):
    S_conf = alpha * SimCLIP(T_hat, I_hat) + (1 - alpha) * SimSem(T_hat, A)
Candidates below threshold tau=0.75 are discarded; the max-confidence candidate
is kept (Eq. 13). K=5 candidates per sample.

ToyCLIP is a tiny CLIP-style dual encoder (offline) providing a shared text/image
space; it is trained contrastively in train.py.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchors import anchor_ids
from data import IMG_C, STOI, VOCAB
from model import D, ImageEncoder

V = len(VOCAB)
PAD = STOI["<pad>"]


class ToyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_emb = nn.Embedding(V, D, padding_idx=PAD)
        self.img_enc = ImageEncoder()
        self.tproj = nn.Linear(D, D)
        self.iproj = nn.Linear(D, D)

    def encode_text(self, ids):
        return F.normalize(self.tproj(self.text_emb(ids).mean(1)), dim=-1)

    def encode_image(self, img):
        return F.normalize(self.iproj(self.img_enc(img).mean(1)), dim=-1)

    def clip_loss(self, ids, img, temp=0.1):
        t = self.encode_text(ids)
        v = self.encode_image(img)
        logits = t @ v.t() / temp
        labels = torch.arange(t.size(0))
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _pad(seqs: List[List[int]]):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])


@torch.no_grad()
def sim_clip(clip: ToyCLIP, text_ids, image):
    """Cross-modal consistency: cosine(text, image) in CLIP space."""
    return (clip.encode_text(text_ids) * clip.encode_image(image)).sum(-1)


@torch.no_grad()
def sim_sem(clip: ToyCLIP, text_ids, anchor_ids_b):
    """Anchor fidelity: cosine(generated text, structured anchor string)."""
    return (clip.encode_text(text_ids) * clip.encode_text(anchor_ids_b)).sum(-1)


@torch.no_grad()
def dual_constraint_select(clip, cand_text_ids: List[List[int]], cand_images,
                           anchor_id_list: List[int], alpha=0.6, tau=0.75):
    """Score K candidates (Eq. 12), drop < tau, return best index + score (Eq. 13).

    Returns (best_idx or None, scores_list)."""
    text = _pad(cand_text_ids)
    anchors = _pad([anchor_id_list] * len(cand_text_ids))
    s_clip = sim_clip(clip, text, cand_images)
    s_sem = sim_sem(clip, text, anchors)
    # map cos(-1,1) -> (0,1) so the tau=0.75 threshold matches the paper's scale
    s_clip = (s_clip + 1) / 2
    s_sem = (s_sem + 1) / 2
    sconf = alpha * s_clip + (1 - alpha) * s_sem
    keep = (sconf >= tau)
    scores = sconf.tolist()
    if keep.sum() == 0:
        return None, scores
    masked = sconf.clone()
    masked[~keep] = -1
    return int(torch.argmax(masked)), scores
