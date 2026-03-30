from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import issue_to_onehot, load_examples
from model import Ranker
from patches import acceptance_oracle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    examples = load_examples(args.test)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = Ranker()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    correct = 0
    accepted = 0

    for i, ex in enumerate(examples):
        mem = torch.tensor(ex.memory.to_vec(), dtype=torch.float32)
        iss = torch.tensor(issue_to_onehot(ex.issue_type), dtype=torch.float32)

        feats = []
        for c in ex.candidates:
            pr = torch.tensor(c.to_vec(), dtype=torch.float32)
            feats.append(torch.cat([mem, iss, pr], dim=0))
        X = torch.stack(feats, dim=0)

        with torch.no_grad():
            scores = model(X)
            pred = int(scores.argmax().item())

        correct += 1 if pred == ex.label else 0
        accepted += 1 if acceptance_oracle(ex.memory, ex.candidates[pred]) else 0

        if i < args.show:
            pr = ex.candidates[pred]
            print(
                f"[{ex.eid}] issue={ex.issue_type} mem={ex.memory} -> chosen PRPlan(has_tests={pr.has_tests}, updates_docs={pr.updates_docs}, diff_lines={pr.diff_lines})"
            )

    n = len(examples)
    print(f"examples={n} top1_acc={correct/max(1,n):.3f} accept_rate={accepted/max(1,n):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
