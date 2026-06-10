import argparse
import json
import os

import numpy as np
import torch

from data import ToyFashionDataset, ToyFashionSpec, pil_to_tensor, split_indices, tokenize_image
from model import AttributeConditionedTransformer, ModelConfig


def compute_map(model, dataset, indices, tokenizer: str, device: str, spec: ToyFashionSpec):
    model.eval()

    all_imgs = [pil_to_tensor(dataset[i]["image"]) for i in indices]
    all_attr_values = np.stack([dataset[i]["attr_values"] for i in indices], axis=0)

    out = {}
    overall_ap = []

    with torch.no_grad():
        for attr_id in range(spec.num_attrs):
            token_list = [tokenize_image(img, tokenizer=tokenizer) for img in all_imgs]
            max_len = max(t.shape[0] for t in token_list)
            token_dim = token_list[0].shape[1]

            tokens = torch.zeros(len(token_list), max_len, token_dim, dtype=torch.float32)
            for i, t in enumerate(token_list):
                tokens[i, : t.shape[0]] = t

            attr_ids = torch.full((len(token_list),), attr_id, dtype=torch.long)
            emb = model(tokens.to(device), attr_ids.to(device)).cpu()
            sim = emb @ emb.t()

            labels = all_attr_values[:, attr_id]
            aps = []
            for i in range(len(indices)):
                scores = sim[i].numpy()
                order = np.argsort(-scores)
                rel = (labels[order] == labels[i]).astype(np.int32)
                rel[order == i] = 0

                num_rel = rel.sum()
                if num_rel == 0:
                    continue

                cumsum = np.cumsum(rel)
                precision_at_k = cumsum / (np.arange(len(rel)) + 1)
                ap = (precision_at_k * rel).sum() / num_rel
                aps.append(float(ap))
                overall_ap.append(float(ap))

            out[f"attr_{attr_id}"] = float(np.mean(aps)) if aps else 0.0

    out["overall"] = float(np.mean(overall_ap)) if overall_ap else 0.0
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, choices=["patch", "superpixel"], default="superpixel")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_items", type=int, default=600)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    spec = ToyFashionSpec()
    dataset = ToyFashionDataset(num_items=args.num_items, spec=spec, seed=args.seed)
    _, test_idx = split_indices(len(dataset), train_ratio=args.train_ratio, seed=args.seed)

    ckpt = torch.load(os.path.join(args.ckpt_dir, "ckpt.pt"), map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"])

    model = AttributeConditionedTransformer(cfg)
    model.load_state_dict(ckpt["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scores = compute_map(model, dataset, test_idx, args.tokenizer, device, spec)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
