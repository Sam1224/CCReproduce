from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import MarketConfig, ToyMarket, build_retrieval_triplets, generate_market, set_seed
from model import DualProfileEncoder, LACRMConfig, LambdaRankReranker, VIEW_NAMES


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def market_to_tensors(market: ToyMarket) -> dict[str, torch.Tensor]:
    return {
        "category": torch.as_tensor(market.category, dtype=torch.long, device=DEVICE),
        "brand": torch.as_tensor(market.brand, dtype=torch.long, device=DEVICE),
        "intent": torch.as_tensor(market.intent, dtype=torch.long, device=DEVICE),
        "style": torch.as_tensor(market.style, dtype=torch.long, device=DEVICE),
        "audience": torch.as_tensor(market.audience, dtype=torch.long, device=DEVICE),
        "price": torch.as_tensor(market.price, dtype=torch.long, device=DEVICE),
        "channel": torch.as_tensor(market.channel, dtype=torch.long, device=DEVICE),
        "tier": torch.as_tensor(market.merchant_tier, dtype=torch.long, device=DEVICE),
    }


def retrieval_loss(
    encoder: DualProfileEncoder,
    fields: dict[str, torch.Tensor],
    src: torch.Tensor,
    view: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> torch.Tensor:
    query_fields = {k: v[src] for k, v in fields.items()}
    q = encoder.encode_queries(query_fields, view)

    all_targets = torch.cat([pos.unsqueeze(1), neg], dim=1)  # [B,1+M]
    tgt_fields = {k: v[all_targets.reshape(-1)] for k, v in fields.items()}
    a = encoder.encode_anchors(tgt_fields).view(all_targets.shape[0], all_targets.shape[1], -1)

    logits = torch.einsum("bd,bkd->bk", q, a) / 0.12
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def build_candidate_bank(
    encoder: DualProfileEncoder,
    fields: dict[str, torch.Tensor],
    market: ToyMarket,
    topk_per_view: int = 20,
):
    anchors = encoder.encode_anchors(fields)                    # [N,D]
    n = anchors.shape[0]
    sem_scores = torch.zeros(n, len(VIEW_NAMES), n, device=DEVICE)
    hit = torch.zeros_like(sem_scores)
    candidates: list[np.ndarray] = []

    for view_idx in range(len(VIEW_NAMES)):
        view_ids = torch.full((n,), view_idx, dtype=torch.long, device=DEVICE)
        queries = encoder.encode_queries(fields, view_ids)
        scores = torch.einsum("id,jd->ij", queries, anchors)
        scores.fill_diagonal_(-1e9)

        if view_idx == len(VIEW_NAMES) - 1:
            inactive = torch.as_tensor(market.festival_active == 0, device=DEVICE)
            scores[inactive] = -1e9

        top_idx = torch.topk(scores, k=min(topk_per_view, n - 1), dim=1).indices
        sem_scores[:, view_idx] = scores
        for src in range(n):
            hit[src, view_idx, top_idx[src]] = 1.0

    for src in range(n):
        pool = torch.nonzero(hit[src].sum(dim=0) > 0, as_tuple=False).squeeze(-1).cpu().numpy()
        pool = pool[pool != src]
        candidates.append(pool.astype(np.int64))

    return sem_scores, hit, candidates


def make_pair_features(
    market: ToyMarket,
    sem_scores: torch.Tensor,
    hit: torch.Tensor,
    candidates: list[np.ndarray],
):
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for src, cand in enumerate(candidates):
        if cand.size == 0:
            features.append(torch.empty(0, 0))
            labels.append(torch.empty(0))
            continue
        sem = sem_scores[src, :, cand].transpose(0, 1).contiguous()  # [C,5]
        h = hit[src, :, cand].transpose(0, 1).contiguous()            # [C,5]

        src_beh = torch.as_tensor(market.behavior[src], dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(len(cand), 1)
        tgt_beh = torch.as_tensor(market.behavior[cand], dtype=torch.float32, device=DEVICE)
        diff = torch.abs(src_beh - tgt_beh)
        feat = torch.cat([sem, h, src_beh, tgt_beh, diff], dim=1)
        lab = torch.as_tensor(market.train_scores[src, cand], dtype=torch.float32, device=DEVICE)
        features.append(feat)
        labels.append(lab)
    return features, labels


def lambdarank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return scores.new_tensor(0.0)
    diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    valid = diff > 1e-6
    if valid.sum() == 0:
        return scores.new_tensor(0.0)
    sdiff = scores.unsqueeze(1) - scores.unsqueeze(0)
    weights = diff.clamp(min=0.0)
    loss = F.softplus(-sdiff) * weights
    return loss[valid].mean()


def main() -> None:
    set_seed(7)
    market_cfg = MarketConfig()
    model_cfg = LACRMConfig(
        n_categories=market_cfg.n_categories,
        n_brands=market_cfg.n_brands,
        n_intents=market_cfg.n_intents,
        n_styles=market_cfg.n_styles,
        n_audiences=market_cfg.n_audiences,
        n_prices=market_cfg.n_prices,
        n_channels=market_cfg.n_channels,
        behavior_dim=market_cfg.behavior_dim,
    )
    market = generate_market(market_cfg)
    fields = market_to_tensors(market)

    encoder = DualProfileEncoder(model_cfg).to(DEVICE)
    triplets = build_retrieval_triplets(market)
    opt = torch.optim.AdamW(encoder.parameters(), lr=3e-3, weight_decay=1e-4)

    retrieval_history = []
    batch_size = 128
    for epoch in range(1, 11):
        encoder.train()
        rng = np.random.default_rng(100 + epoch)
        order = rng.permutation(len(triplets))
        loss_sum = 0.0
        n_steps = 0
        for start in range(0, len(order), batch_size):
            chunk = [triplets[idx] for idx in order[start : start + batch_size]]
            src = torch.as_tensor([x[0] for x in chunk], dtype=torch.long, device=DEVICE)
            view = torch.as_tensor([x[1] for x in chunk], dtype=torch.long, device=DEVICE)
            pos = torch.as_tensor([x[2] for x in chunk], dtype=torch.long, device=DEVICE)
            neg = torch.as_tensor(np.stack([x[3] for x in chunk]), dtype=torch.long, device=DEVICE)
            loss = retrieval_loss(encoder, fields, src, view, pos, neg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach())
            n_steps += 1
        retrieval_history.append({"epoch": epoch, "retrieval_loss": round(loss_sum / max(1, n_steps), 4)})
        print(f"[retrieval] epoch={epoch:02d} loss={loss_sum / max(1, n_steps):.4f}")

    encoder.eval()
    sem_scores, hit, candidates = build_candidate_bank(encoder, fields, market)
    group_features, group_labels = make_pair_features(market, sem_scores, hit, candidates)

    input_dim = next(feat.shape[1] for feat in group_features if feat.numel() > 0)
    reranker = LambdaRankReranker(input_dim=input_dim, hidden_dim=model_cfg.rerank_hidden).to(DEVICE)
    rank_opt = torch.optim.AdamW(reranker.parameters(), lr=2e-3, weight_decay=1e-4)

    rerank_history = []
    for epoch in range(1, 13):
        reranker.train()
        losses = []
        for feat, lab in zip(group_features, group_labels):
            if feat.numel() == 0 or feat.shape[0] <= 1:
                continue
            score = reranker(feat)
            loss = lambdarank_loss(score, lab)
            rank_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            rank_opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        rerank_history.append({"epoch": epoch, "rerank_loss": round(mean_loss, 4)})
        print(f"[rerank]   epoch={epoch:02d} loss={mean_loss:.4f}")

    ckpt = {
        "market_cfg": asdict(market_cfg),
        "model_cfg": asdict(model_cfg),
        "market": {
            "category": market.category,
            "brand": market.brand,
            "intent": market.intent,
            "style": market.style,
            "audience": market.audience,
            "price": market.price,
            "channel": market.channel,
            "merchant_tier": market.merchant_tier,
            "festival_active": market.festival_active,
            "behavior": market.behavior,
            "train_view_scores": market.train_view_scores,
            "eval_view_scores": market.eval_view_scores,
            "train_scores": market.train_scores,
            "eval_scores": market.eval_scores,
        },
        "encoder_state": encoder.state_dict(),
        "reranker_state": reranker.state_dict(),
    }
    torch.save(ckpt, ROOT / "lacrm_ckpt.pt")

    metrics = {
        "device": str(DEVICE),
        "num_advertisers": int(market_cfg.n_advertisers),
        "num_triplets": int(len(triplets)),
        "candidate_mean": round(float(np.mean([len(c) for c in candidates])), 2),
        "retrieval_history": retrieval_history,
        "rerank_history": rerank_history,
    }
    (ROOT / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {ROOT / 'lacrm_ckpt.pt'}")
    print(f"saved: {ROOT / 'train_metrics.json'}")


if __name__ == "__main__":
    main()

