from __future__ import annotations

import argparse

import torch

from captioner import RuleBasedCaptioner, should_route_to_captioner
from dataset import build_catalog, generate_queries, iter_titles
from model import (
    ProductEncoder,
    Vocabulary,
    captionfuse_query,
    encode_products,
    encode_titles,
    rank_targets,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/toy_encoder.pt")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(path: str, device: str) -> tuple[ProductEncoder, Vocabulary]:
    checkpoint = torch.load(path, map_location=device)
    vocab = Vocabulary([])
    vocab.itos = checkpoint["vocab"]
    vocab.stoi = {token: i for i, token in enumerate(vocab.itos)}
    model = ProductEncoder(len(vocab), checkpoint["num_products"], dim=checkpoint["dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab


@torch.no_grad()
def main() -> None:
    args = parse_args()
    catalog = build_catalog()
    queries = generate_queries(catalog, n_queries=args.queries)
    model, vocab = load_model(args.checkpoint, args.device)
    captioner = RuleBasedCaptioner()

    product_ids = [p.product_id for p in catalog]
    gallery = encode_titles(model, vocab, list(iter_titles(catalog)), args.device)
    image_vectors = encode_products(model, product_ids, args.device)

    raw_query_vectors = []
    captionfuse_vectors = []
    hybrid_vectors = []
    target_ids = []
    routed = 0
    for query in queries:
        ref = catalog[query.reference_id]
        target_ids.append(query.target_id)

        raw_text_vec = model.encode_text_ids(vocab.encode(query.modification).to(args.device))
        raw_query_vectors.append(raw_text_vec)

        caption = captioner.generate(ref, query.modification, query.changed_attrs)
        caption_vec = model.encode_text_ids(vocab.encode(caption).to(args.device))
        fused = captionfuse_query(image_vectors[query.reference_id], caption_vec, alpha=args.alpha)
        captionfuse_vectors.append(fused)

        if should_route_to_captioner(query.modification):
            routed += 1
            hybrid_vectors.append(fused)
        else:
            hybrid_vectors.append(captionfuse_query(image_vectors[query.reference_id], raw_text_vec, alpha=args.alpha))

    results = {
        "raw_sparse_text": summarize_metrics(rank_targets(torch.stack(raw_query_vectors), gallery, target_ids)),
        "captionfuse": summarize_metrics(rank_targets(torch.stack(captionfuse_vectors), gallery, target_ids)),
        "hybrid_routing": summarize_metrics(rank_targets(torch.stack(hybrid_vectors), gallery, target_ids)),
    }
    print(f"queries={len(queries)} alpha={args.alpha} routed_to_captioner={routed / len(queries):.2%}")
    for name, metrics in results.items():
        print(name)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
