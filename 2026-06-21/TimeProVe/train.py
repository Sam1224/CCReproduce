"""Train TimeProVe on the toy data (CPU, a few seconds).

Two independently-supervised modules (as in the paper, ACE detector and the
verifier are separate stages):
  1. Action Detector  -- multi-label BCE on per-segment action activations (Eq. 1).
  2. Temporal Verifier-- BCE on (clip, query, candidate-answer) -> supports? labels,
     built from the ranked hypotheses of the proposal generator (Eqs. 3-9) over the
     GOLD timeline (teacher forcing), so the verifier learns to accept the clip that
     truly grounds the gold answer and reject the rest.

Saves timeprove.pt for test.py.
"""
from __future__ import annotations

import torch

from data import ACTIONS, split
from model import (ActionDetector, TemporalVerifier, TimeProVe,
                   proposal_generator)


def iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def verifier_label(sample, ans, w) -> float:
    if sample.gold_window is None:
        return 0.0
    if sample.tau == "STATE":
        return 1.0 if (ans == "yes" and iou(w, sample.gold_window) > 0.0) else 0.0
    return 1.0 if (ans == sample.gold_answer and iou(w, sample.gold_window) > 0.5) else 0.0


def train_detector(train, epochs=120, lr=5e-3):
    det = ActionDetector()
    opt = torch.optim.Adam(det.parameters(), lr=lr)
    # one (feats, seg_labels) per distinct video is enough; QA pairs share videos
    seen = {}
    for s in train:
        seen[id(s.feats)] = (s.feats, s.seg_labels)
    vids = list(seen.values())
    for ep in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for feats, labels in vids:
            logits = det(feats)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels)
        loss = loss / len(vids)
        loss.backward()
        opt.step()
        if ep % 30 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc = []
                for feats, labels in vids:
                    pred = (det.probs(feats) >= 0.5).float()
                    acc.append((pred == labels).float().mean().item())
                print(f"[Detector] epoch {ep:3d}  bce={loss.item():.4f}  "
                      f"seg-acc={sum(acc)/len(acc):.3f}")
    return det


def train_verifier(train, epochs=120, lr=5e-3):
    ver = TemporalVerifier()
    opt = torch.optim.Adam(ver.parameters(), lr=lr)
    # build (sample, answer, window, label) examples from GOLD-timeline proposals
    examples = []
    for s in train:
        for ans, w, _ in proposal_generator(s.events, s.q_tokens, s.tau, s.L):
            examples.append((s, ans, w, verifier_label(s, ans, w)))
    pos = sum(1 for *_, y in examples if y > 0.5)
    print(f"[Verifier] {len(examples)} examples ({pos} positive)")
    for ep in range(epochs):
        opt.zero_grad()
        logits, labels = [], []
        for s, ans, w, y in examples:
            logits.append(ver(s.feats, w, s.q_tokens, ans))
            labels.append(y)
        logits = torch.stack(logits)
        labels = torch.tensor(labels)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
        if ep % 30 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()
                print(f"[Verifier] epoch {ep:3d}  bce={loss.item():.4f}  acc={acc:.3f}")
    return ver


def main():
    torch.manual_seed(0)
    train, _ = split()
    print(f"train QA pairs: {len(train)}")
    det = train_detector(train)
    ver = train_verifier(train)
    model = TimeProVe(det, ver)
    torch.save(model.state_dict(), "timeprove.pt")
    print("saved -> timeprove.pt")


if __name__ == "__main__":
    main()
