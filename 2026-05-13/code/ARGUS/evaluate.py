"""
ARGUS evaluation script.

Evaluates policy-adaptive performance across policy versions.

Usage:
    python evaluate.py --model_dir outputs/ --data data/toy_ad_governance.jsonl
"""

import argparse
import json
import torch
from collections import defaultdict
from models.ad_classifier import AdClassifier, LABEL_TO_IDX, LABEL_SPACE


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate_model(model, data, label_key="new_label"):
    model.eval()
    results = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for sample in data:
            gold = sample.get(label_key)
            if gold not in LABEL_TO_IDX:
                continue
            pred, conf = model.predict(sample["text"])
            policy_tag = sample.get("policy_tag", "none")
            results["overall"]["total"] += 1
            results[policy_tag]["total"] += 1
            if pred == gold:
                results["overall"]["correct"] += 1
                results[policy_tag]["correct"] += 1

    return {
        k: v["correct"] / max(v["total"], 1)
        for k, v in results.items()
    }


def evaluate(model_dir, data_path):
    all_data = load_jsonl(data_path)
    test_data = [d for d in all_data if d.get("split") == "test"]

    print(f"\n{'='*55}")
    print("  ARGUS POLICY-ADAPTIVE EVALUATION")
    print(f"{'='*55}")
    print(f"  Test samples: {len(test_data)}")

    model_names = ["model_stage1.pt", "model_stage2.pt", "model_final.pt"]

    for model_name in model_names:
        model_path = f"{model_dir}/{model_name}"
        try:
            model = AdClassifier()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
        except FileNotFoundError:
            print(f"\n  [{model_name}] Not found — skipping.")
            continue

        accs = evaluate_model(model, test_data)

        stage_label = {
            "model_stage1.pt": "Stage I  (Policy Seeding)",
            "model_stage2.pt": "Stage II (PDU Rectify)",
            "model_final.pt": "Stage III (Final)",
        }.get(model_name, model_name)

        print(f"\n  [{stage_label}]")
        print(f"  {'Tag':<25} {'Accuracy':>10}")
        print(f"  {'-'*36}")
        for tag, acc in sorted(accs.items(), key=lambda x: x[0]):
            print(f"  {tag:<25} {acc:>10.4f}")

    print(f"\n  Paper reports: Stage II triggers significant performance leap;")
    print(f"  Stage III further improves Historical Recall on gray-area samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs")
    parser.add_argument("--data", default="data/toy_ad_governance.jsonl")
    args = parser.parse_args()
    evaluate(args.model_dir, args.data)
