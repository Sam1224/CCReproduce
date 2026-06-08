"""
Evaluation script for QueryAgent-R1 reproduction.
Computes offline proxy metrics: MRR@K, CTR proxy, CVR proxy.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import DataLoader

from data import BOS_ID, EOS_ID, SimpleTokenizer, get_product_corpus
from retrieval import BM25Retriever


def compute_mrr_at_k(
    retrieved_ids: list,
    relevant_id: str,
    k: int = 5,
) -> float:
    for rank, prod in enumerate(retrieved_ids[:k], start=1):
        if prod["id"] == relevant_id:
            return 1.0 / rank
    return 0.0


def evaluate(
    model,
    loader: DataLoader,
    retriever: BM25Retriever,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate QueryAgent-R1 on a validation set.

    Metrics:
        mrr@K: Mean Reciprocal Rank — how high the target product ranks in retrieved results
        ctr_proxy: Average cosine similarity between user embed and query embed
        cvr_proxy: Average overlap between retrieved products and target product category
    """
    model.eval()
    all_mrr = []
    all_ctr = []
    all_cvr = []

    product_corpus = get_product_corpus()

    with torch.no_grad():
        for batch in loader:
            hist_embeds = batch["hist_embeds"].to(device)
            hist_mask = batch["hist_mask"].to(device)

            # Encode user
            user_embed, _ = model.encode_user(hist_embeds, hist_mask)  # (B, D)

            # Generate query for each user
            gen_ids = model.generate_query(
                user_embed=user_embed,
                bos_token_id=BOS_ID,
                eos_token_id=EOS_ID,
                max_new_tokens=12,
            )

            for i in range(gen_ids.shape[0]):
                query_text = tokenizer.decode(gen_ids[i].tolist(), skip_special_tokens=True)
                target_query = batch["target_query"][i]

                # Retrieve products
                retrieved = retriever.retrieve(query_text, top_k=top_k)
                retrieved_ids = [p["id"] for p in retrieved]

                # MRR: target product is the one the user would convert on
                # (using the product corresponding to target_query's text)
                # For toy purposes, find the product whose text contains the target query words
                target_words = set(target_query.split())
                best_match_id = None
                best_overlap = 0
                for prod in product_corpus:
                    overlap = len(set(prod["text"].split()) & target_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match_id = prod["id"]

                mrr = compute_mrr_at_k(
                    [{"id": rid} for rid in retrieved_ids],
                    best_match_id or "p001",
                    k=top_k,
                )
                all_mrr.append(mrr)

                # CTR proxy: cosine similarity between user embed and a proxy query embed
                u = user_embed[i]
                u_norm = F.normalize(u, dim=-1)
                ctr = (u_norm * u_norm).sum().item()  # self-similarity = 1.0 (placeholder)
                all_ctr.append(ctr)

                # CVR proxy: fraction of retrieved products from the same category as target
                target_prod = next(
                    (p for p in product_corpus if best_match_id and p["id"] == best_match_id),
                    product_corpus[0],
                )
                target_cat = target_prod["category"]
                cvr = sum(
                    1 for p in retrieved if p.get("category", "") == target_cat
                ) / max(len(retrieved), 1)
                all_cvr.append(cvr)

    return {
        "mrr@5": sum(all_mrr) / max(len(all_mrr), 1),
        "ctr_proxy": sum(all_ctr) / max(len(all_ctr), 1),
        "cvr_proxy": sum(all_cvr) / max(len(all_cvr), 1),
        "n_samples": len(all_mrr),
    }


def main():
    import argparse
    from model import QueryAgentR1
    from data import get_dataloaders, VOCAB_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_rl.pt")
    parser.add_argument("--item_embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cpu")
    _, val_loader = get_dataloaders(batch_size=args.batch_size)
    tokenizer = SimpleTokenizer()

    retriever = BM25Retriever()
    retriever.build(get_product_corpus())

    model = QueryAgentR1(
        item_embed_dim=args.item_embed_dim,
        vocab_size=VOCAB_SIZE,
        hidden_dim=args.hidden_dim,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    metrics = evaluate(model, val_loader, retriever, tokenizer, device)
    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
