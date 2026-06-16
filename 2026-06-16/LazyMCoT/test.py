"""Evaluate LazyMCoT on the toy test set.

Reports:
  - accuracy of LazyMCoT vs the Direct (single-pass) baseline
  - fraction short-circuited (direct) vs grounded
  - average "latency" proxy = mean number of VLM forward passes
demonstrating the accuracy / efficiency trade-off.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from data import ToyConfig, build_splits, set_seed
from model import (CollaborativeGrounding, LazyMCoT, VisualExpertSAM3,
                   VLMInterface, compute_first_token_stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", type=str, default="router.joblib")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = ToyConfig()
    _cal, test = build_splits(cfg, seed=args.seed)

    blob = joblib.load(args.router)
    router, s_floor, alpha = blob["router"], blob["s_floor"], blob["alpha"]
    router.s_floor = s_floor

    vlm = VLMInterface()
    expert = VisualExpertSAM3()
    cg = CollaborativeGrounding(vlm, expert)
    system = LazyMCoT(vlm, router, cg)

    n = len(test)
    correct_lazy = correct_direct = 0
    n_grounded = 0
    total_fwd = 0
    # confusion: did grounding fix originally-wrong samples?
    fixed = broke = 0

    for s in test:
        out = system.predict(s)
        total_fwd += out["n_forward"]
        if out["mode"] == "grounded":
            n_grounded += 1
        lazy_ok = out["pred"] == s.answer_id
        direct_ok = out["direct_pred"] == s.answer_id
        correct_lazy += int(lazy_ok)
        correct_direct += int(direct_ok)
        if out["mode"] == "grounded":
            if (not direct_ok) and lazy_ok:
                fixed += 1
            if direct_ok and (not lazy_ok):
                broke += 1

    acc_lazy = correct_lazy / n
    acc_direct = correct_direct / n
    frac_grounded = n_grounded / n

    print("=" * 60)
    print(f"LazyMCoT toy evaluation  (n_test={n}, alpha={alpha}, s_floor={s_floor:.4f})")
    print("=" * 60)
    print(f"Direct (single-pass) accuracy : {acc_direct:.3f}")
    print(f"LazyMCoT accuracy             : {acc_lazy:.3f}   (delta {acc_lazy-acc_direct:+.3f})")
    print(f"Short-circuited (direct)      : {(1-frac_grounded):.3f}")
    print(f"Routed to grounding (CG)      : {frac_grounded:.3f}")
    print(f"Avg VLM forward passes / sample (latency proxy): {total_fwd/n:.2f}")
    print(f"   (Direct=1 pass; CG adds entity+attention+LPD passes)")
    print(f"Grounding fixed wrong->right  : {fixed}")
    print(f"Grounding broke right->wrong  : {broke}")
    print("=" * 60)


if __name__ == "__main__":
    main()
