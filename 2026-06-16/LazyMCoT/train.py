"""Train the LazyMCoT router on the held-out calibration split D_cal.

Steps:
  1) run the base VLM (single pass) over D_cal to collect (topp, delta_logit, y)
  2) fit the GBDT router g_theta
  3) conformal calibration: out-of-fold scores -> alpha-quantile over the
     must-recall (ori-wrong) subset -> s_floor
  4) save {router, s_floor, alpha} to disk
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.model_selection import cross_val_predict

from data import ToyConfig, build_splits, collect_routing_dataset, set_seed
from model import AdaptiveRouter, VLMInterface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="conformal level; ~(1-alpha) hard-sample recall (smaller=more conservative)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="router.joblib")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ToyConfig()
    cal, _test = build_splits(cfg, seed=args.seed)
    vlm = VLMInterface()

    print(f"[1/4] Collecting first-token stats over D_cal (n={len(cal)}) ...")
    X, y, info = collect_routing_dataset(cal, vlm)
    print(f"      ori-acc={info['ori_acc']:.3f}  #wrong(must-recall)={info['n_wrong']}/{info['n']}")

    print("[2/4] Out-of-fold scores for conformal calibration ...")
    probe = AdaptiveRouter()
    oof_p = cross_val_predict(probe.model, X, y, cv=5, method="predict_proba")[:, 1]
    oof_scores = AdaptiveRouter._logit(oof_p)

    print("[3/4] Fitting GBDT router g_theta on full D_cal ...")
    router = AdaptiveRouter().fit(X, y)

    scores_mr = oof_scores[y == 1]  # must-recall subset D_mr
    s_floor = router.calibrate_threshold(scores_mr, args.alpha)
    print(f"      alpha={args.alpha}  s_floor={s_floor:.4f}")

    # routing stats on D_cal (using in-sample scores for a quick sanity check)
    s_all = router.score(X)
    routed = s_all >= s_floor
    recall = float((routed & (y == 1)).sum() / max(1, (y == 1).sum()))
    frac_routed = float(routed.mean())
    print(f"[4/4] D_cal routing: routed={frac_routed:.3f}  hard-recall={recall:.3f}")

    joblib.dump({"router": router, "s_floor": s_floor, "alpha": args.alpha,
                 "feature_names": ["topp", "delta_logit"]}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
