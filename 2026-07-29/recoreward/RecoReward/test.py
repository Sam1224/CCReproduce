import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToyLiveStreamDataset, collate_streams
from model import ContentPolicy


def recall_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=(10, 64, 128)):
    metrics = {}
    ranking = torch.argsort(scores, dim=1, descending=True)
    for k in ks:
        topk = ranking[:, : min(k, ranking.size(1))]
        hit = (topk == labels[:, None]).any(dim=1).float()
        metrics[f"HR@{k}"] = hit.mean().item()
        ndcg = []
        for row, label in zip(topk, labels):
            match = (row == label).nonzero(as_tuple=False)
            ndcg.append(0.0 if match.numel() == 0 else 1.0 / torch.log2(match[0, 0].float() + 2.0).item())
        metrics[f"NDCG@{k}"] = sum(ndcg) / len(ndcg)
    ranks = (ranking == labels[:, None]).nonzero(as_tuple=False)[:, 1].float() + 1.0
    metrics["MRR"] = (1.0 / ranks).mean().item()
    return metrics


def evaluate(args):
    dataset = ToyLiveStreamDataset(num_items=args.num_items, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_streams)
    policy = ContentPolicy()
    if args.checkpoint:
        policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    policy.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            description = policy(batch["content"])["description"]
            item_bank = F.normalize(batch["item_bank"], dim=-1)
            all_scores.append(description @ item_bank.T)
            all_labels.append(batch["positive_item"])
    metrics = recall_metrics(torch.cat(all_scores), torch.cat(all_labels))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num-items", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    evaluate(parser.parse_args())
