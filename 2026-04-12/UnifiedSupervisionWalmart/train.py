from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import ToySponsoredDataset, build_target_distribution, build_toy_dataset
from metrics import avg_relevance_at_k, ndcg_at_k, precision_at_k
from model import BiEncoderRetriever


def _pad(tokens: List[int], length: int) -> List[int]:
    return tokens + [0] * (length - len(tokens))


def _to_tensor(token_seqs: List[List[int]], device: torch.device) -> torch.Tensor:
    max_len = max(len(x) for x in token_seqs)
    padded = [_pad(x, max_len) for x in token_seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate(
    *,
    dataset: ToySponsoredDataset,
    model: BiEncoderRetriever,
    device: torch.device,
    k: int = 25,
) -> Dict[str, float]:
    model.eval()

    item_tokens_all = _to_tensor(dataset.item_tokens, device)

    p_list: List[float] = []
    ndcg_list: List[float] = []
    avgrel_list: List[float] = []
    avgen_list: List[float] = []

    for ex in dataset.test:
        q = _to_tensor([ex.query_tokens], device)
        cand_ids = torch.tensor(ex.candidate_item_ids, dtype=torch.long, device=device)
        cand_tokens = item_tokens_all[cand_ids]

        q_vec = model.encode_query(q).repeat(cand_tokens.shape[0], 1)
        d_vec = model.encode_item(cand_tokens)
        scores = model.score(q_vec, d_vec).detach().cpu().numpy()

        ranked_idx = list(np.argsort(-scores))
        ranked_rels = [ex.graded_relevance[i] for i in ranked_idx]
        ranked_eng = [ex.engagement_score[i] for i in ranked_idx]

        p_list.append(precision_at_k(ranked_rels, k))
        ndcg_list.append(ndcg_at_k(ranked_rels, k))
        avgrel_list.append(avg_relevance_at_k(ranked_rels, k))
        avgen_list.append(float(np.mean(ranked_eng[:k])))

    return {
        f"P@{k}": float(np.mean(p_list)),
        f"NDCG@{k}": float(np.mean(ndcg_list)),
        f"AvgRel@{k}": float(np.mean(avgrel_list)),
        f"AvgEng@{k}": float(np.mean(avgen_list)),
    }


def train_one(
    *,
    dataset: ToySponsoredDataset,
    mode: str,
    device: torch.device,
    epochs: int,
    lr: float,
    dim: int,
    temperature: float,
) -> BiEncoderRetriever:
    model = BiEncoderRetriever(vocab_size=dataset.vocab_size, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    item_tokens_all = _to_tensor(dataset.item_tokens, device)

    for _ in range(epochs):
        model.train()
        for ex in dataset.train:
            q = _to_tensor([ex.query_tokens], device)  # [1, T]
            cand_ids = torch.tensor(ex.candidate_item_ids, dtype=torch.long, device=device)
            cand_tokens = item_tokens_all[cand_ids]

            q_vec = model.encode_query(q).repeat(cand_tokens.shape[0], 1)
            d_vec = model.encode_item(cand_tokens)
            logits = model.score(q_vec, d_vec) / temperature  # [C]

            target = build_target_distribution(ex, mode=mode, temperature=1.0)
            target_t = torch.tensor(target, dtype=torch.float32, device=device)

            log_pred = torch.log_softmax(logits, dim=0)
            loss = torch.sum(target_t * (torch.log(target_t + 1e-12) - log_pred))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")

    dataset = build_toy_dataset(seed=args.seed)

    os.makedirs("checkpoints", exist_ok=True)

    results = {}
    for mode in ["eng_only", "rel_only", "rel_eng"]:
        model = train_one(
            dataset=dataset,
            mode=mode,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            dim=args.dim,
            temperature=args.temperature,
        )
        torch.save({"state": model.state_dict(), "meta": {"vocab_size": dataset.vocab_size, "dim": args.dim}}, f"checkpoints/{mode}.pt")
        metrics = evaluate(dataset=dataset, model=model, device=device)
        results[mode] = metrics

    print("=== Offline evaluation (toy) ===")
    for mode in ["eng_only", "rel_only", "rel_eng"]:
        m = results[mode]
        print(
            f"{mode:8s}  P@25={m['P@25']:.4f}  NDCG@25={m['NDCG@25']:.4f}  "
            f"AvgRel@25={m['AvgRel@25']:.4f}  AvgEng@25={m['AvgEng@25']:.4f}"
        )


if __name__ == "__main__":
    main()
