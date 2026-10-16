from __future__ import annotations

import argparse
import json

import torch

from data import build_expanded_index, build_title_only_index, build_toy_dataset, evaluate_index
from model import ModelConfig, SAMD2QModel, decode_top_terms


def load_model(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**checkpoint["cfg"])
    model = SAMD2QModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/sam_d2q/model.pt")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    dataset = build_toy_dataset(seed=args.seed)
    model, checkpoint = load_model(args.ckpt)

    with torch.no_grad():
        probs = model.predict_probs(
            dataset.masked_title_tokens,
            dataset.masked_title_masks,
            dataset.image_tensors,
        )
    decoded = decode_top_terms(probs, dataset.id_to_token, topk=5, threshold=0.30)
    generated_terms = {product.product_id: decoded[idx] for idx, product in enumerate(dataset.products)}

    title_only = evaluate_index(
        dataset.eval_queries,
        dataset.products,
        build_title_only_index(dataset.products),
        topk=1,
    )
    expanded = evaluate_index(
        dataset.eval_queries,
        dataset.products,
        build_expanded_index(dataset.products, generated_terms),
        topk=1,
    )

    print(
        json.dumps(
            {
                "title_only": title_only,
                "expanded": expanded,
                "saved_baseline": checkpoint.get("baseline"),
                "saved_full": checkpoint.get("full_metrics"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
