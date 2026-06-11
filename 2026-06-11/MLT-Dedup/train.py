import argparse
import json
import os

import numpy as np

from data import ToyVideoConfig, build_pairs, generate_videos
from models import ToyDiFSiM, ToyMultiLevelEncoder


def f1_precision_recall(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/mlt_dedup")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score_thr_min", type=float, default=0.30)
    parser.add_argument("--score_thr_max", type=float, default=0.90)
    parser.add_argument("--score_thr_steps", type=int, default=61)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = ToyVideoConfig()
    videos, dup_meta = generate_videos(cfg, seed=args.seed)
    pairs = build_pairs(videos, dup_meta, cfg, seed=args.seed)

    n = len(pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]

    encoder = ToyMultiLevelEncoder(clip_stride=4)
    matcher = ToyDiFSiM(diff_weight=0.5, cell_thr=0.65)

    # pre-encode frame embeddings
    frame_emb = np.stack([encoder.encode_frame(v) for v in videos], axis=0)

    def eval_pairs(ps, score_thr: float):
        y_true = []
        y_pred = []
        for p in ps:
            a = frame_emb[p["a"]]
            b = frame_emb[p["b"]]
            out = matcher.match(a, b)
            pred = 1 if out["score"] >= score_thr and out["overlap_ratio"] >= 0.15 else 0
            y_true.append(int(p["label"]))
            y_pred.append(pred)
        y_true = np.array(y_true, dtype=np.int32)
        y_pred = np.array(y_pred, dtype=np.int32)
        prec, rec, f1 = f1_precision_recall(y_true, y_pred)
        return prec, rec, f1

    best = {"score_thr": 0.6, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    thresholds = np.linspace(args.score_thr_min, args.score_thr_max, args.score_thr_steps)
    for thr in thresholds:
        prec, rec, f1 = eval_pairs(val_pairs, score_thr=float(thr))
        if f1 > best["f1"]:
            best = {"score_thr": float(thr), "f1": float(f1), "precision": float(prec), "recall": float(rec)}

    print("best on val:", best)

    out = {
        "toy_cfg": cfg.__dict__,
        "matcher": {"diff_weight": matcher.diff_weight, "cell_thr": matcher.cell_thr, "score_thr": best["score_thr"], "min_overlap_ratio": 0.15},
        "dup_meta": dup_meta,
    }

    with open(os.path.join(args.out_dir, "ckpt.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved: {os.path.join(args.out_dir, 'ckpt.json')}")


if __name__ == "__main__":
    main()
