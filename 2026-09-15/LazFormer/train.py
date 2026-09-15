from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DataConfig, PretrainSeqDataset, RankDataset, collate_pretrain, collate_rank, recall_ndcg
from model import LazFormer, ModelConfig

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
PRETRAIN_CKPT = CKPT_DIR / "lazformer_pretrain.pt"
LAZFORMER_CKPT = CKPT_DIR / "lazformer_rank.pt"
BASELINE_CKPT = CKPT_DIR / "baseline_rank.pt"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_rank(
    model,
    loader,
    device: torch.device,
    ks=(5, 10),
    *,
    truncate_last: Optional[int] = None,
) -> Dict[str, float]:
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


def train_pretrain(cfg: DataConfig, mcfg: ModelConfig, args: argparse.Namespace, device: torch.device) -> None:
    ds = PretrainSeqDataset(cfg, domain="source")
    loader = DataLoader(ds, batch_size=args.pretrain_batch, shuffle=True, collate_fn=collate_pretrain)

    model = LazFormer(mcfg, use_adapter=False, topm_history=cfg.topm_history).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-2)

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in loader:
            req = batch["req"].to(device)
            seq = batch["seq"].to(device)
            lengths = batch["lengths"].to(device)

            loss = model.pretrain_loss(req, seq, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            n += 1

        print(f"[pretrain] epoch {epoch:02d}  loss={loss_sum / max(1, n):.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "pretrain",
            "model_cfg": mcfg.__dict__,
            "data_cfg": cfg.__dict__,
            "state_dict": model.state_dict(),
        },
        PRETRAIN_CKPT,
    )
    print(f"[pretrain] saved: {PRETRAIN_CKPT}")


def train_rank(
    cfg: DataConfig,
    mcfg: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
    *,
    baseline_only: bool = False,
) -> None:
    tr = RankDataset(cfg, split="train", domain="target")
    va = RankDataset(cfg, split="valid", domain="target")

    tr_loader = DataLoader(tr, batch_size=args.rank_batch, shuffle=True, collate_fn=collate_rank)
    va_loader = DataLoader(va, batch_size=args.rank_batch * 2, shuffle=False, collate_fn=collate_rank)

    # ---- baseline: vanilla Transformer ranker from scratch ----
    # dense attention, no pretrain, no adapters
    baseline = LazFormer(
        mcfg,
        use_adapter=False,
        topm_history=cfg.topm_history,
        use_sparse=False,
    ).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=args.rank_lr, weight_decay=1e-3)

    best_b = None
    best_b_state = None

    for epoch in range(1, args.rank_epochs + 1):
        baseline.train()
        for batch in tr_loader:
            req = batch["req"].to(device)
            hist = batch["hist"].to(device)
            lengths = batch["lengths"].to(device)
            cands = batch["cands"].to(device)
            label = batch["label"].to(device)

            # baseline truncates to last-M interactions (typical maxlen practice)
            if cfg.topm_history < hist.shape[1]:
                hist = hist[:, -cfg.topm_history :]
                lengths = lengths.clamp_max(cfg.topm_history)
            logits = baseline.rank_logits(req, hist, lengths, cands)
            loss = F.cross_entropy(logits, label)

            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            opt_b.step()

        val = evaluate_rank(baseline, va_loader, device=device, truncate_last=cfg.topm_history)
        key = val["NDCG@10"]
        if best_b is None or key > best_b:
            best_b = key
            best_b_state = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
        print(
            f"[baseline(trunc={cfg.topm_history})] epoch {epoch:02d}  val R@10={val['Recall@10']:.4f}  NDCG@10={val['NDCG@10']:.4f}"
        )

    torch.save(
        {
            "stage": "rank",
            "model": "baseline_truncated_transformer",
            "model_cfg": mcfg.__dict__,
            "baseline_trunc": cfg.topm_history,
            "data_cfg": cfg.__dict__,
            "state_dict": best_b_state if best_b_state is not None else baseline.state_dict(),
        },
        BASELINE_CKPT,
    )
    print(f"[baseline] saved: {BASELINE_CKPT}")

    if baseline_only:
        return

    # ---- LazFormer: load pretrained backbone + train adapters for transfer ----
    if not PRETRAIN_CKPT.exists():
        raise SystemExit(f"missing {PRETRAIN_CKPT}; run pretrain first")

    pre = torch.load(PRETRAIN_CKPT, weights_only=False)

    laz = LazFormer(mcfg, use_adapter=True, topm_history=cfg.topm_history).to(device)
    laz.load_state_dict(pre["state_dict"], strict=False)
    laz.freeze_backbone_for_rank()

    trainable = [p for p in laz.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.rank_lr, weight_decay=1e-3)

    best = None
    best_state = None

    for epoch in range(1, args.rank_epochs + 1):
        laz.train()
        for batch in tr_loader:
            req = batch["req"].to(device)
            hist = batch["hist"].to(device)
            lengths = batch["lengths"].to(device)
            cands = batch["cands"].to(device)
            label = batch["label"].to(device)

            logits = laz.rank_logits(req, hist, lengths, cands)
            loss = F.cross_entropy(logits, label)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        val = evaluate_rank(laz, va_loader, device=device)
        key = val["NDCG@10"]
        if best is None or key > best:
            best = key
            best_state = {k: v.detach().cpu().clone() for k, v in laz.state_dict().items()}
        print(f"[lazformer] epoch {epoch:02d}  val R@10={val['Recall@10']:.4f}  NDCG@10={val['NDCG@10']:.4f}")

    torch.save(
        {
            "stage": "rank",
            "model": "lazformer",
            "model_cfg": mcfg.__dict__,
            "data_cfg": cfg.__dict__,
            "state_dict": best_state if best_state is not None else laz.state_dict(),
        },
        LAZFORMER_CKPT,
    )
    print(f"[lazformer] saved: {LAZFORMER_CKPT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)

    parser.add_argument("--rank-epochs", type=int, default=8)
    parser.add_argument("--rank-batch", type=int, default=64)
    parser.add_argument("--rank-lr", type=float, default=2e-3)

    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--adapter-dim", type=int, default=24)

    parser.add_argument("--only-rank", action="store_true")
    parser.add_argument("--only-baseline", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg = DataConfig(seed=args.seed)
    mcfg = ModelConfig(
        num_items=cfg.num_items,
        num_requests=cfg.num_requests,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        adapter_dim=args.adapter_dim,
    )

    if not args.only_rank:
        train_pretrain(cfg, mcfg, args, device)

    train_rank(cfg, mcfg, args, device, baseline_only=args.only_baseline)


if __name__ == "__main__":
    main()
