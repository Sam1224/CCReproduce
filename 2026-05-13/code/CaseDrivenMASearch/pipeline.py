"""
Full closed-loop pipeline for Case-Driven Multi-Agent E-Commerce Search Relevance.

Usage:
    python pipeline.py --data data/toy_ecom_pairs.jsonl --rounds 3
"""

import argparse
import json
import os
import random
import torch
from torch.optim import AdamW

from agents.user_agent import UserAgent
from agents.annotator_agent import AnnotatorAgent
from agents.optimizer_agent import OptimizerAgent, BadCase
from models.generative_relevance_model import GenerativeRelevanceModel, LABEL_SPACE


def load_pairs(path: str) -> list:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def run_pipeline(data_path: str, rounds: int = 3, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    all_pairs = load_pairs(data_path)
    train_pairs = [p for p in all_pairs if p.get("split") == "train"]
    test_pairs  = [p for p in all_pairs if p.get("split") == "test"]
    print(f"Loaded {len(train_pairs)} train / {len(test_pairs)} test pairs.")

    # ── Initialize model ───────────────────────────────────────────────────────
    model = GenerativeRelevanceModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    label2idx = {l: i for i, l in enumerate(LABEL_SPACE)}

    # ── Initialize agents ─────────────────────────────────────────────────────
    user_agent = UserAgent(relevance_threshold="substitute")
    annotator_agent = AnnotatorAgent(num_paths=3, temperature=1.0)
    optimizer_agent = OptimizerAgent(training_pairs=train_pairs)

    current_train = list(train_pairs)

    for round_idx in range(rounds):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_idx + 1} / {rounds}")
        print(f"{'='*60}")

        # ── Step 1: Train GRM on current data ──────────────────────────────────
        print("\n[Step 1] Training Generative Relevance Model...")
        model.train()
        for epoch in range(5):
            random.shuffle(current_train)
            total_loss = 0.0
            for pair in current_train:
                gold_label = pair.get("label")
                if gold_label not in label2idx:
                    continue
                label_tensor = torch.tensor([label2idx[gold_label]])
                out = model.forward(
                    [pair["query"]],
                    [pair["product_title"]],
                    [pair["product_desc"]],
                    labels=label_tensor,
                )
                optimizer.zero_grad()
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
            avg_loss = total_loss / max(len(current_train), 1)
            print(f"  Epoch {epoch+1}/5 | Loss: {avg_loss:.4f}")

        # ── Step 2: User Agent — discover bad cases ────────────────────────────
        print("\n[Step 2] User Agent discovering bad cases...")
        interactions = user_agent.find_bad_cases(current_train)
        report = user_agent.report(interactions)
        print(f"  Bad case rate: {report['bad_case_rate']:.2%}")

        # ── Step 3: Annotator Agent — re-label bad cases ───────────────────────
        bad_pairs = [
            {"query": i.query, "product_title": i.product_title, "product_desc": i.product_desc}
            for i in interactions if i.is_bad_case
        ]
        if bad_pairs:
            print(f"\n[Step 3] Annotator Agent re-labeling {len(bad_pairs)} bad cases...")
            anno_results = annotator_agent.annotate_batch(bad_pairs)
            gold_labels = [
                i.gold_label for i in interactions if i.is_bad_case
            ]
            precision = annotator_agent.compute_precision(anno_results, gold_labels)
            print(f"  Annotator precision: {precision:.4f}")

            # Save annotations
            anno_out = []
            for r in anno_results:
                anno_out.append({
                    "query": r.query,
                    "product_title": r.product_title,
                    "final_label": r.final_label,
                    "confidence": r.confidence,
                    "rationale": r.rationale,
                })
            with open(f"{output_dir}/annotations_round{round_idx+1}.jsonl", "w") as f:
                for item in anno_out:
                    f.write(json.dumps(item) + "\n")

            # ── Step 4: Optimizer Agent — root cause + resolution ────────────────
            print(f"\n[Step 4] Optimizer Agent resolving {len(bad_pairs)} bad cases...")
            bad_case_objs = [
                BadCase(
                    query=i.query,
                    product_title=i.product_title,
                    product_desc=i.product_desc,
                    predicted_label=i.predicted_label,
                    gold_label=i.gold_label,
                    bad_case_reason=i.bad_case_reason,
                )
                for i in interactions if i.is_bad_case
            ]
            resolutions = optimizer_agent.resolve_batch(bad_case_objs)
            synthetic_data = optimizer_agent.collect_synthetic_data(resolutions)

            if synthetic_data:
                print(f"  Adding {len(synthetic_data)} synthetic samples to training set.")
                current_train.extend(synthetic_data)

            # Save resolutions
            with open(f"{output_dir}/resolutions_round{round_idx+1}.jsonl", "w") as f:
                for r in resolutions:
                    f.write(json.dumps({
                        "root_cause": r.root_cause.value,
                        "diagnosis": r.diagnosis,
                        "action": r.action,
                    }) + "\n")

        # ── Step 5: Evaluate on test set ──────────────────────────────────────
        print(f"\n[Step 5] Evaluating on {len(test_pairs)} test pairs...")
        model.eval()
        correct = 0
        with torch.no_grad():
            for pair in test_pairs:
                pred, conf = model.predict(
                    pair["query"], pair["product_title"], pair["product_desc"]
                )
                if pred == pair.get("label"):
                    correct += 1
        acc = correct / max(len(test_pairs), 1)
        print(f"  Test Accuracy: {acc:.4f}")

    print("\nPipeline complete. Outputs saved to:", output_dir)
    torch.save(model.state_dict(), f"{output_dir}/grm_final.pt")
    print("Model saved to:", f"{output_dir}/grm_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_ecom_pairs.jsonl")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    run_pipeline(args.data, args.rounds, args.output_dir)
