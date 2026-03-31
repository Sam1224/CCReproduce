from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from dataset import SUBSETS, make_dataset, split
from model import RAIRModel


def macro_f1(y: np.ndarray, p: np.ndarray, n_classes: int = 4) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def acc_at_2(arr: np.ndarray) -> np.ndarray:
    return (arr >= 2).astype(np.int64)


def eval_split(model: RAIRModel, examples, subset: str) -> dict:
    ys = []
    ps = []
    for ex in examples:
        if ex.subset != subset:
            continue
        logits = model(
            torch.tensor([ex.q_cat]),
            torch.tensor([ex.q_brand]),
            torch.tensor([ex.q_color]),
            torch.tensor([ex.item_cat]),
            torch.tensor([ex.item_brand]),
            torch.tensor([ex.item_color_text]),
            torch.tensor([ex.item_color_img]),
            ex.rule_ids.unsqueeze(0),
        )
        pred = int(torch.argmax(logits, dim=-1).item())
        ys.append(ex.y)
        ps.append(pred)

    y = np.array(ys, dtype=np.int64)
    p = np.array(ps, dtype=np.int64)

    return {
        "n": int(len(y)),
        "acc4": float((y == p).mean()) if len(y) else 0.0,
        "acc2": float((acc_at_2(y) == acc_at_2(p)).mean()) if len(y) else 0.0,
        "macro_f1": macro_f1(y, p) if len(y) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_rules", action="store_true")
    args = ap.parse_args()

    data = make_dataset(n=6000, seed=8)
    _, te = split(data, 0.0)

    model = RAIRModel(use_rules=args.use_rules)
    ckpt = Path("checkpoints/rair.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)

    model.eval()

    print(f"use_rules={args.use_rules}")
    for subset in SUBSETS:
        m = eval_split(model, te, subset)
        print(f"[{subset}] n={m['n']} acc@4={m['acc4']:.3f} acc@2={m['acc2']:.3f} macro-f1={m['macro_f1']:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
