import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from data import ToyFashionDataset, ToyFashionSpec, pil_to_tensor, set_seed, split_indices, tokenize_image
from model import AttributeConditionedTransformer, ModelConfig


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    emb = F.normalize(emb, dim=-1)
    sim = emb @ emb.t()
    sim = sim / temperature

    labels = labels.view(-1, 1)
    mask = labels.eq(labels.t())
    mask.fill_diagonal_(False)

    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~torch.eye(emb.size(0), dtype=torch.bool, device=emb.device))

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss


def build_batch(dataset: ToyFashionDataset, indices: np.ndarray, attr_id: int, tokenizer: str, device: str):
    images = [pil_to_tensor(dataset[i]["image"]) for i in indices]
    labels = torch.tensor([int(dataset[i]["attr_values"][attr_id]) for i in indices], dtype=torch.long)

    token_list = [tokenize_image(img, tokenizer=tokenizer) for img in images]
    max_len = max(t.shape[0] for t in token_list)
    token_dim = token_list[0].shape[1]

    tokens = torch.zeros(len(token_list), max_len, token_dim, dtype=torch.float32)
    for i, t in enumerate(token_list):
        tokens[i, : t.shape[0]] = t

    attr_ids = torch.full((len(token_list),), attr_id, dtype=torch.long)

    return tokens.to(device), attr_ids.to(device), labels.to(device)


def evaluate_map(model, dataset, indices, tokenizer: str, device: str, spec: ToyFashionSpec) -> float:
    model.eval()

    all_imgs = [pil_to_tensor(dataset[i]["image"]) for i in indices]
    all_attr_values = np.stack([dataset[i]["attr_values"] for i in indices], axis=0)

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
                overall_ap.append(float(ap))

    model.train()
    return float(np.mean(overall_ap)) if overall_ap else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, choices=["patch", "superpixel"], default="superpixel")
    parser.add_argument("--out_dir", type=str, default="runs/superfashion")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_items", type=int, default=600)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    spec = ToyFashionSpec()
    dataset = ToyFashionDataset(num_items=args.num_items, spec=spec, seed=args.seed)
    train_idx, test_idx = split_indices(len(dataset), train_ratio=args.train_ratio, seed=args.seed)

    token_dim = 6 if args.tokenizer == "patch" else 5
    cfg = ModelConfig(token_dim=token_dim, num_attrs=spec.num_attrs)
    model = AttributeConditionedTransformer(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    history = []
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            attr_id = int(rng.integers(0, spec.num_attrs))
            idx = rng.choice(train_idx, size=args.batch_size, replace=False)
            tokens, attr_ids, labels = build_batch(dataset, idx, attr_id, args.tokenizer, device)

            emb = model(tokens, attr_ids)
            loss = supervised_contrastive_loss(emb, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        map_score = evaluate_map(model, dataset, test_idx, args.tokenizer, device, spec)
        history.append({"epoch": epoch, "mAP": map_score})
        print(f"epoch={epoch:02d} tokenizer={args.tokenizer} mAP={map_score:.4f}")

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "spec": asdict(spec), "tokenizer": args.tokenizer}, ckpt_path)

    with open(os.path.join(args.out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
