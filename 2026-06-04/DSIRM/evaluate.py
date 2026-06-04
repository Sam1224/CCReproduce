from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import GroupEval, build_toy_dataset
from model import DualEncoder, QuerySidPredictor, SimpleTokenizer
from ranker import SidRanker
from rqvae import ResidualVectorQuantizer
from sid import prefix_match_features


def _dcg(rels: List[int]) -> float:
    out = 0.0
    for i, r in enumerate(rels, start=1):
        out += (2**r - 1) / np.log2(i + 1)
    return float(out)


def ndcg_at_k(labels: List[int], scores: List[float], k: int = 10) -> float:
    order = np.argsort(-np.asarray(scores))[:k]
    rels = [labels[i] for i in order]
    ideal = sorted(labels, reverse=True)[:k]
    denom = _dcg(ideal)
    if denom <= 0:
        return 0.0
    return _dcg(rels) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=str, default="runs/stage1.pt")
    parser.add_argument("--stage2", type=str, default="runs/stage2.pt")
    parser.add_argument("--stage3", type=str, default="runs/stage3.pt")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = torch.load(args.stage1, map_location="cpu")
    stage2 = torch.load(args.stage2, map_location="cpu")
    stage3 = torch.load(args.stage3, map_location="cpu")

    cfg = stage1["config"]
    hidden = int(cfg["hidden"])
    num_codebooks = int(cfg["codebooks"])
    codebook_size = int(cfg["codebook_size"])

    tokenizer = SimpleTokenizer()
    tokenizer.vocab = stage1["tokenizer_vocab"]

    encoder = DualEncoder(vocab_size=len(tokenizer.vocab), hidden_size=hidden)
    encoder.load_state_dict(stage1["encoder_state"])
    encoder.to(device)
    encoder.eval()

    quantizer = ResidualVectorQuantizer(dim=hidden, num_codebooks=num_codebooks, codebook_size=codebook_size)
    quantizer.load_state_dict(stage1["quantizer_state"])
    quantizer.to(device)
    quantizer.eval()

    predictor = QuerySidPredictor(
        vocab_size=len(tokenizer.vocab),
        hidden_size=hidden,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )
    predictor.load_state_dict(stage2["predictor_state"])
    predictor.to(device)
    predictor.eval()

    ranker = SidRanker(num_codebooks=num_codebooks)
    ranker.load_state_dict(stage3["ranker_state"])
    ranker.to(device)
    ranker.eval()

    bundle = build_toy_dataset()

    all_y = []
    all_p = []
    ndcgs = []

    for g in bundle.eval_groups:
        q_batch = tokenizer.encode_batch([g.query])
        q_batch.token_ids = q_batch.token_ids.to(device)
        q_batch.attn_mask = q_batch.attn_mask.to(device)

        d_batch = tokenizer.encode_batch(g.candidate_items)
        d_batch.token_ids = d_batch.token_ids.to(device)
        d_batch.attn_mask = d_batch.attn_mask.to(device)

        with torch.no_grad():
            q_emb = encoder.encode(q_batch.token_ids, q_batch.attn_mask)  # (1,H)
            d_emb = encoder.encode(d_batch.token_ids, d_batch.attn_mask)  # (K,H)
            d_codes = quantizer(d_emb).codes

            q_logits = predictor(q_batch)  # (1,L,K)
            q_codes = torch.argmax(q_logits, dim=-1).repeat(d_emb.shape[0], 1)

            dense_sim = torch.sum(q_emb.repeat(d_emb.shape[0], 1) * d_emb, dim=-1, keepdim=True)
            prefix = prefix_match_features(q_codes, d_codes)
            feats = torch.cat([dense_sim, prefix], dim=1)

            scores = torch.sigmoid(ranker(feats)).detach().cpu().numpy().tolist()

        ndcgs.append(ndcg_at_k(g.labels, scores, k=args.k))

        all_y.extend(g.labels)
        all_p.extend(scores)

    auc = roc_auc_score(all_y, all_p)
    print(f"AUC={auc:.4f}  NDCG@{args.k}={float(np.mean(ndcgs)):.4f}  (groups={len(ndcgs)})")


if __name__ == "__main__":
    main()
