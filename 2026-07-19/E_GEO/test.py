import argparse

import torch

from data import rewrite_description
from model import GEOReranker, PromptMetaOptimizer, rank_improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/egeo_ranker.pt")
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ranker = GEOReranker()
    ranker.load_state_dict(ckpt["ranker"])
    meta = PromptMetaOptimizer()
    meta.logits.data.copy_(ckpt["meta_logits"])

    query = "durable travel bottle for hiking and daily commuting"
    original = "steel bottle with cap"
    best, scores = rank_improvement(ranker, query, original)
    meta_choice = meta.choose()
    print({
        "query": query,
        "best_scored_rewrite": best,
        "meta_optimizer_choice": meta_choice,
        "rewritten_description": rewrite_description(original, meta_choice),
        "heuristic_scores": [round(float(x), 4) for x in scores.detach()],
    })


if __name__ == "__main__":
    main()
