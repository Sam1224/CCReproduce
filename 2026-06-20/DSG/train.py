"""Calibrate the semantic-cache threshold tau on the toy set.

tau gates Algorithm 1's semantic-cache reuse: cos(e(q), e(q_j)) >= tau.
We sweep tau and pick the value maximising balanced accuracy on:
  * positives: paraphrase pairs that SHOULD reuse the cached result (same answer)
  * negatives: distinct queries that should NOT reuse (different answer)
Result is written to tau.json and consumed by test.py.
"""
from __future__ import annotations

import json

import numpy as np

from data import calibration_pairs
from embeddings import cosine, embed


def sweep():
    pos, neg = calibration_pairs()
    pos_scores = [cosine(embed(a.text), embed(b.text)) for a, b in pos]
    neg_scores = [cosine(embed(a.text), embed(b.text)) for a, b in neg]

    best_tau, best_acc = 0.5, -1.0
    for tau in np.linspace(0.30, 0.99, 70):
        tp = sum(s >= tau for s in pos_scores)
        tn = sum(s < tau for s in neg_scores)
        acc = 0.5 * (tp / len(pos_scores) + tn / len(neg_scores))   # balanced acc
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau)
    return best_tau, best_acc, pos_scores, neg_scores


def main():
    tau, acc, pos_scores, neg_scores = sweep()
    print("=== DSG semantic-cache tau calibration ===")
    print(f"positive (paraphrase) pairs : {len(pos_scores)}  "
          f"mean cos={np.mean(pos_scores):.3f}")
    print(f"negative (distinct)   pairs : {len(neg_scores)}  "
          f"mean cos={np.mean(neg_scores):.3f}")
    print(f"selected tau = {tau:.3f}  (balanced calibration acc = {acc:.3f})")
    with open("tau.json", "w") as f:
        json.dump({"tau": round(tau, 4), "calibration_acc": round(acc, 4)}, f, indent=2)
    print("saved -> tau.json")


if __name__ == "__main__":
    main()
