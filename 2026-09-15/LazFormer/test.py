from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data import DataConfig, RankDataset, build_cfg_from_ckpt, collate_rank, recall_ndcg
from model import LazFormer, ModelConfig
from train import BASELINE_CKPT, LAZFORMER_CKPT


@torch.no_grad()
def eval_model(model, loader, device: torch.device, ks=(5, 10), *, truncate_last: Optional[int] = None):
    model.eval()
    agg = {f"Recall@{k}": 0.0 for k in ks}
    agg.update({f"NDCG@{k}": 0.0 for k in ks})
    n = 0
    for batch in loader:
        req = batch["req"].to(device)
        hist = batch["hist"].to(device)
        lengths = batch["lengths"].to(device)
        cands = batch["cands"].to(device)
        label = batch["label"].to(device)

        if truncate_last is not None and truncate_last < hist.shape[1]:
            hist = hist[:, -truncate_last:]
            lengths = lengths.clamp_max(truncate_last)

        scores = model.rank_logits(req, hist, lengths, cands)
        m = recall_ndcg(scores, label, ks=ks)
        for k, v in m.items():
            agg[k] += v
        n += 1
    for k in list(agg.keys()):
        agg[k] /= max(1, n)
    return agg


def main() -> None:
    if not os.path.exists(BASELINE_CKPT) or not os.path.exists(LAZFORMER_CKPT):
        raise SystemExit("missing checkpoints; please run `python train.py` first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck_b = torch.load(BASELINE_CKPT, weights_only=False)
    ck_l = torch.load(LAZFORMER_CKPT, weights_only=False)

    cfg: DataConfig = build_cfg_from_ckpt(ck_l)

    test_ds = RankDataset(cfg, split="test", domain="target")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_rank)

    # baseline: vanilla Transformer ranker (dense attention, trained from scratch)
    mcfg_b = ModelConfig(**ck_b["model_cfg"])
    baseline = LazFormer(mcfg_b, use_adapter=False, topm_history=cfg.topm_history, use_sparse=False)
    baseline.load_state_dict(ck_b["state_dict"], strict=True)
    baseline.to(device)

    # lazformer
    mcfg = ModelConfig(**ck_l["model_cfg"])
    laz = LazFormer(mcfg, use_adapter=True, topm_history=cfg.topm_history)
    laz.load_state_dict(ck_l["state_dict"], strict=True)
    laz.to(device)

    trunc = ck_b.get("baseline_trunc", None)
    mb = eval_model(baseline, test_loader, device, truncate_last=trunc)
    ml = eval_model(laz, test_loader, device)

    print("=== Target-domain ranking metrics (1 pos + N neg in candidate set) ===")
    print("baseline :", {k: round(v, 4) for k, v in mb.items()})
    print("lazformer:", {k: round(v, 4) for k, v in ml.items()})


if __name__ == "__main__":
    main()
