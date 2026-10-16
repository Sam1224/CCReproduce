from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Sequence

import torch

from data import ToyDataset, build_toy_dataset, evaluate_index, build_title_only_index, build_expanded_index
from model import ModelConfig, SAMD2QModel, decode_top_terms, grpo_loss, multilabel_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def take_examples(dataset: ToyDataset, split: str):
    return {
        "train": dataset.train_examples,
        "val": dataset.val_examples,
        "test": dataset.test_examples,
    }[split]


def prepare_batch(dataset: ToyDataset, product_ids: Sequence[int], use_masked_title: bool) -> Dict[str, torch.Tensor]:
    index = torch.tensor(product_ids, dtype=torch.long)
    title_tokens = dataset.masked_title_tokens[index] if use_masked_title else dataset.title_tokens[index]
    title_masks = dataset.masked_title_masks[index] if use_masked_title else dataset.title_masks[index]
    return {
        "title_tokens": title_tokens,
        "title_masks": title_masks,
        "images": dataset.image_tensors[index],
        "targets": dataset.target_vectors[index],
        "visual_terms": dataset.visual_term_vectors[index],
        "commerce_weights": torch.tensor(
            [dataset.products[int(pid)].commerce_weight for pid in product_ids],
            dtype=torch.float32,
        ),
    }


def evaluate_model(model: SAMD2QModel, dataset: ToyDataset, use_masked_title: bool) -> Dict[str, float]:
    model.eval()
    title_tokens = dataset.masked_title_tokens if use_masked_title else dataset.title_tokens
    title_masks = dataset.masked_title_masks if use_masked_title else dataset.title_masks
    probs = model.predict_probs(title_tokens, title_masks, dataset.image_tensors)
    decoded = decode_top_terms(probs, dataset.id_to_token, topk=5, threshold=0.30)
    generated = {product.product_id: decoded[idx] for idx, product in enumerate(dataset.products)}
    expanded_index = build_expanded_index(dataset.products, generated)
    return evaluate_index(dataset.eval_queries, dataset.products, expanded_index, topk=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stage1_epochs", type=int, default=120)
    parser.add_argument("--stage2_epochs", type=int, default=80)
    parser.add_argument("--stage3_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--out_dir", type=str, default="runs/sam_d2q")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_toy_dataset(seed=args.seed)
    cfg = ModelConfig(vocab_size=len(dataset.vocab), pad_id=dataset.vocab["<pad>"])
    model = SAMD2QModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_ids = [example.product_id for example in dataset.train_examples]

    history: List[Dict[str, float]] = []
    best_quality = -1.0
    best_state = None
    best_stage = ""

    def maybe_update_best(stage_name: str, metrics: Dict[str, float]) -> None:
        nonlocal best_quality, best_state, best_stage
        metric_value = float(metrics.get("TotalQuality@1", 0.0))
        if metric_value > best_quality:
            best_quality = metric_value
            best_stage = stage_name
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "cfg": asdict(cfg),
                "vocab": dataset.vocab,
                "id_to_token": dataset.id_to_token,
            }

    def run_supervised_stage(stage_name: str, epochs: int, use_masked_title: bool) -> None:
        for epoch in range(1, epochs + 1):
            model.train()
            batch = prepare_batch(dataset, train_ids, use_masked_title=use_masked_title)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["title_tokens"], batch["title_masks"], batch["images"])
            loss = multilabel_loss(logits, batch["targets"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == epochs:
                metrics = evaluate_model(model, dataset, use_masked_title=use_masked_title)
                maybe_update_best(stage_name, metrics)
                record = {"stage": stage_name, "epoch": epoch, "loss": float(loss.item()), **metrics}
                history.append(record)
                print(json.dumps(record))

    run_supervised_stage("stage1_mm_sft", args.stage1_epochs, use_masked_title=False)
    run_supervised_stage("stage2_counterfactual", args.stage2_epochs, use_masked_title=True)

    for epoch in range(1, args.stage3_epochs + 1):
        model.train()
        batch = prepare_batch(dataset, train_ids, use_masked_title=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        samples, log_probs = model.sample_term_sets(
            batch["title_tokens"],
            batch["title_masks"],
            batch["images"],
            vocab_terms=dataset.id_to_token,
            topk=5,
            samples_per_item=4,
        )
        loss = grpo_loss(
            log_prob_sums=log_probs,
            sampled_terms=samples,
            gold_terms=batch["targets"],
            visual_terms=batch["visual_terms"],
            commerce_weights=batch["commerce_weights"],
            vocab_map=dataset.vocab,
        )
        anchor_logits = model(batch["title_tokens"], batch["title_masks"], batch["images"])
        anchor_loss = multilabel_loss(anchor_logits, batch["targets"])
        loss = 0.7 * loss + 0.3 * anchor_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.stage3_epochs:
            metrics = evaluate_model(model, dataset, use_masked_title=True)
            maybe_update_best("stage3_business_alignment", metrics)
            record = {"stage": "stage3_business_alignment", "epoch": epoch, "loss": float(loss.item()), **metrics}
            history.append(record)
            print(json.dumps(record))

    baseline = evaluate_index(dataset.eval_queries, dataset.products, build_title_only_index(dataset.products), topk=1)
    assert best_state is not None
    model.load_state_dict(best_state["model"])
    full_metrics = evaluate_model(model, dataset, use_masked_title=True)

    probs = model.predict_probs(dataset.masked_title_tokens.to(device), dataset.masked_title_masks.to(device), dataset.image_tensors.to(device))
    decoded = decode_top_terms(probs, dataset.id_to_token, topk=5, threshold=0.30)
    generated_terms = {product.product_id: decoded[idx] for idx, product in enumerate(dataset.products)}

    torch.save(
        {
            **best_state,
            "generated_terms": generated_terms,
            "baseline": baseline,
            "full_metrics": full_metrics,
            "selected_stage": best_stage,
        },
        os.path.join(args.out_dir, "model.pt"),
    )
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "generated_terms.json"), "w", encoding="utf-8") as f:
        json.dump(generated_terms, f, indent=2)

    print(json.dumps({"baseline": baseline, "full": full_metrics}, indent=2))


if __name__ == "__main__":
    main()
