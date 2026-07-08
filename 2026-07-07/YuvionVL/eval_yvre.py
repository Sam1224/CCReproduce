"""
YuvionVL — YVRE (Yuvion VL RiskEval) Evaluation Script

Implements the three-level progressive adversarial evaluation framework:
  Level 1: Open-source general benchmarks
  Level 2: Open-source safety benchmarks
  Level 3: In-house capability & e-commerce business benchmarks
             (logo recognition, brand knock-off, product category, price compliance)

Usage:
    python eval_yvre.py --level all
    python eval_yvre.py --level 3
"""

import argparse
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data import YVREDataset, YVREItem
from model import YuvionVLModel, SafetyGate


# ─── Simple text tokenizer ────────────────────────────────────────────────────

def tokenize(text: str, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [min(ord(c), 31999) for c in text[:max_len]]
    pad = max_len - len(ids)
    return (
        torch.tensor(ids + [0] * pad, dtype=torch.long).unsqueeze(0),
        torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long).unsqueeze(0),
    )


# ─── YVRE Evaluator ───────────────────────────────────────────────────────────

class YVREEvaluator:
    """
    YVRE three-level evaluation framework.

    In the paper:
    - The model generates free-text reasoning + structured verdict
    - Evaluation uses both exact-match and LLM-judged correctness
    - Level 3 business benchmarks are proprietary and use internal metrics

    In this reproduction:
    - We simulate binary classification (safe/violation) as a proxy
    - Key metrics: accuracy, F1, breakdown by domain and difficulty
    """

    def __init__(self, model: YuvionVLModel, threshold: float = 0.4):
        self.gate = SafetyGate(model, threshold)

    @torch.no_grad()
    def evaluate_level(self, level: int | None = None) -> dict:
        dataset = YVREDataset(level=level)
        if len(dataset) == 0:
            return {"level": level, "num_items": 0, "message": "No items for this level"}

        results_by_domain: dict[str, list[dict]] = defaultdict(list)
        results_by_difficulty: dict[str, list[bool]] = defaultdict(list)

        for item in dataset:
            input_ids, attention_mask = tokenize(item["query"])
            verdicts = self.gate.check(input_ids, attention_mask)
            verdict = verdicts[0]

            # Ground truth: anything containing "违规" or "仿冒" or "异常" → violation
            is_gt_violation = any(kw in item["ground_truth"] for kw in ["违规", "仿冒", "异常", "风险", "是"])
            is_pred_violation = verdict["is_violation"]

            correct = (is_gt_violation == is_pred_violation)
            results_by_domain[item["domain"]].append({
                "correct": correct,
                "confidence": verdict["confidence"],
                "gt_violation": is_gt_violation,
                "pred_violation": is_pred_violation,
            })
            results_by_difficulty[item["difficulty"]].append(correct)

        # Aggregate
        all_correct = [r["correct"] for res in results_by_domain.values() for r in res]
        overall_acc = sum(all_correct) / max(len(all_correct), 1)

        domain_acc = {}
        for domain, res_list in results_by_domain.items():
            domain_acc[domain] = sum(r["correct"] for r in res_list) / max(len(res_list), 1)

        difficulty_acc = {}
        for diff, corrects in results_by_difficulty.items():
            difficulty_acc[diff] = sum(corrects) / max(len(corrects), 1)

        return {
            "level": level,
            "num_items": len(all_correct),
            "overall_accuracy": overall_acc,
            "domain_accuracy": domain_acc,
            "difficulty_accuracy": difficulty_acc,
        }

    def run_full_evaluation(self) -> dict:
        """Run all three YVRE levels and aggregate results."""
        print("=" * 60)
        print("YVRE (Yuvion VL RiskEval) — Three-Level Evaluation")
        print("=" * 60)

        all_results = {}
        level_scores = []

        for level in [1, 2, 3]:
            print(f"\n── Level {level} {'(Open General)' if level==1 else '(Safety)' if level==2 else '(E-commerce Business)'}")
            result = self.evaluate_level(level)

            if result.get("num_items", 0) == 0:
                print(f"   No items available for Level {level} (toy dataset).")
                result["overall_accuracy"] = 0.0
            else:
                print(f"   Items: {result['num_items']}")
                print(f"   Overall accuracy: {result['overall_accuracy']:.4f}")

                if "domain_accuracy" in result:
                    print("   Domain breakdown:")
                    for domain, acc in result["domain_accuracy"].items():
                        print(f"     {domain:30s}: {acc:.4f}")

                if "difficulty_accuracy" in result:
                    print("   Difficulty breakdown:")
                    for diff, acc in result["difficulty_accuracy"].items():
                        print(f"     {diff:12s}: {acc:.4f}")

                level_scores.append(result["overall_accuracy"])

            all_results[f"level_{level}"] = result

        if level_scores:
            overall_yvre_score = sum(level_scores) / len(level_scores)
            all_results["yvre_composite"] = overall_yvre_score
            print(f"\n── YVRE Composite Score: {overall_yvre_score:.4f}")
            print("   (Paper reports +9.9 pts over comparable open-source models,")
            print("    +6.7 pts over larger closed-source models on full YVRE)")

        return all_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YuvionVL YVRE Evaluation")
    parser.add_argument("--level", default="all", choices=["all", "1", "2", "3"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Load model
    model = YuvionVLModel(embed_dim=args.embed_dim, visual_depth=2, text_layers=2)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint provided — using randomly initialized model (for interface demo)")

    model.eval()
    evaluator = YVREEvaluator(model)

    if args.level == "all":
        results = evaluator.run_full_evaluation()
    else:
        level = int(args.level)
        result = evaluator.evaluate_level(level)
        results = {f"level_{level}": result}
        print(f"Level {level} result: {result}")

    print("\nEvaluation complete.")
    return results


if __name__ == "__main__":
    main()
