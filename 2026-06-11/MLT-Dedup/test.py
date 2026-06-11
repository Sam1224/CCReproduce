import argparse
import json
import os

import numpy as np

from data import ToyVideoConfig, build_pairs, generate_videos
from models import ToyDiFSiM, ToyMultiLevelEncoder, cosine_topk


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
    parser.add_argument("--ckpt_dir", type=str, default="runs/mlt_dedup")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    cfg = ToyVideoConfig(**ckpt["toy_cfg"])
    mcfg = ckpt["matcher"]

    videos, dup_meta = generate_videos(cfg, seed=args.seed)
    pairs = build_pairs(videos, dup_meta, cfg, seed=args.seed)

    n = len(pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    test_pairs = pairs[n_train + n_val :]

    encoder = ToyMultiLevelEncoder(clip_stride=4)
    matcher = ToyDiFSiM(diff_weight=mcfg["diff_weight"], cell_thr=mcfg["cell_thr"])

    frame_emb = np.stack([encoder.encode_frame(v) for v in videos], axis=0)
    clip_emb = np.stack([encoder.encode_clip(v) for v in videos], axis=0)

    # Retrieval Recall@5 for duplicates: is the true source in top-5 by clip similarity?
    hits = 0
    for m in dup_meta:
        dup_id = int(m["dup_id"])
        src_id = int(m["src_id"])
        idx = cosine_topk(clip_emb[dup_id], clip_emb, k=5)
        if src_id in set(int(x) for x in idx.tolist()):
            hits += 1
    recall5 = hits / max(1, len(dup_meta))

    y_true = []
    y_pred = []
    overlap_err = []

    for p in test_pairs:
        out = matcher.match(frame_emb[p["a"]], frame_emb[p["b"]])
        pred = 1 if out["score"] >= mcfg["score_thr"] and out["overlap_ratio"] >= mcfg["min_overlap_ratio"] else 0

        y_true.append(int(p["label"]))
        y_pred.append(pred)

        if int(p["label"]) == 1:
            overlap_err.append(abs(out["overlap_ratio"] - float(p["overlap_ratio"])))

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    prec, rec, f1 = f1_precision_recall(y_true, y_pred)

    mae = float(np.mean(overlap_err)) if overlap_err else 0.0

    print(f"Retrieval  | Recall@5 = {recall5:.4f} (dup source in top-5)")
    print(f"Matching  | Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print(f"Overlap   | MAE(overlap_ratio) = {mae:.4f} (positive pairs)")


if __name__ == "__main__":
    main()
