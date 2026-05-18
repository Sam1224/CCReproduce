"""
Evaluation script for MoMoE.

Reproduces the paper's evaluation setup:
- Test on unseen subreddits (cross-community generalization)
- Report Micro-F1, Precision, Recall
- Compare MoMoE-Community vs MoMoE-NormVio
"""

import argparse
import json
import logging
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from momoe import MoMoE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Toy test data for demo evaluation
TOY_TEST_DATA = [
    {"text": "You are the stupidest person I've ever seen. Just shut up.", "label": 1},
    {"text": "I disagree with your point, but let's discuss it calmly.", "label": 0},
    {"text": "Scientists have proven that vaccines cause autism. Here's the study.", "label": 1},
    {"text": "A new peer-reviewed study in Nature shows COVID-19 vaccine safety data.", "label": 0},
    {"text": "Click here for free iPhone! Limited time offer!", "label": 1},
    {"text": "What's the best gaming mouse for competitive FPS games?", "label": 0},
    {"text": "People from [country] are all criminals and should be banned.", "label": 1},
    {"text": "I think immigration policy should be reformed. Here's my reasoning...", "label": 0},
    {"text": "Buy these supplements for instant weight loss, guaranteed results!", "label": 1},
    {"text": "Eating a balanced diet and exercising regularly helps with fitness goals.", "label": 0},
]


def load_test_data(data_path: str) -> list[dict]:
    """Load test data from JSONL file."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate(model: MoMoE, test_data: list[dict]) -> dict:
    """Run evaluation and compute metrics."""
    texts = [d["text"] for d in test_data]
    true_labels = [d["label"] for d in test_data]

    outputs = model.predict(texts)
    pred_labels = [o.label for o in outputs]

    metrics = {
        "micro_f1": f1_score(true_labels, pred_labels, average="micro", zero_division=0),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "n_samples": len(test_data),
        "n_violations": sum(true_labels),
    }

    return metrics, outputs


def print_metrics(metrics: dict, mode: str) -> None:
    print(f"\n{'='*55}")
    print(f"  MoMoE-{mode.title()} Evaluation Results")
    print("="*55)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("="*55)


def main():
    parser = argparse.ArgumentParser(description="MoMoE Evaluation")
    parser.add_argument("--mode", choices=["community", "norm_violation", "both"], default="community")
    parser.add_argument("--data", default=None, help="Path to test JSONL (uses toy data if None)")
    parser.add_argument("--aggregation", choices=["weighted", "majority_vote"], default="weighted")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", default="momoe_results.json")
    parser.add_argument("--explain", action="store_true", help="Generate explanations (requires OpenAI key)")
    args = parser.parse_args()

    test_data = load_test_data(args.data) if args.data else TOY_TEST_DATA
    logger.info(f"Loaded {len(test_data)} test samples")

    modes = ["community", "norm_violation"] if args.mode == "both" else [args.mode]
    all_results = {}

    for mode in modes:
        logger.info(f"Evaluating MoMoE-{mode.title()}...")
        model = MoMoE(
            mode=mode,
            top_k=args.top_k,
            aggregation=args.aggregation,
            use_explainer=args.explain,
        )

        metrics, outputs = evaluate(model, test_data)
        print_metrics(metrics, mode)
        all_results[mode] = metrics

        # Show sample explanation if requested
        if args.explain and outputs:
            o = outputs[0]
            if o.explanation:
                print(f"\nSample Explanation for: '{o.text[:60]}...'")
                print(json.dumps(o.explanation, indent=2, ensure_ascii=False))

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
