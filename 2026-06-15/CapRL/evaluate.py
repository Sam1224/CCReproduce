"""
CapRL++ — Evaluation

Evaluates caption quality using the Prism-style MCQ accuracy metric.

Paper (§3 - Evaluation):
    Primary metric: Prism captioning accuracy (MCQ-based)
    Evaluated on 20+ image and video benchmarks.

Our toy: MCQ accuracy on held-out test set, comparing:
    - SFT baseline (constant skill_level=0.3)
    - CapRL++ trained model (higher skill_level from training)
"""
import json
import os
import sys


def evaluate_model(test_data: list[dict], skill_level: float, scorer) -> dict:
    from captioner import ToyLVLMCaptioner
    captioner = ToyLVLMCaptioner(skill_level=skill_level)

    accuracies = []
    for scene in test_data:
        caption = captioner.generate_caption(scene)
        r = scorer.score_caption(caption, scene.get("mcqs", []))
        accuracies.append(r["accuracy"])

    return {
        "mean_accuracy": sum(accuracies) / (len(accuracies) + 1e-8),
        "n_scenes": len(accuracies),
        "skill_level": skill_level,
    }


def main(
    test_path: str = "data/test.json",
    checkpoint_path: str = "checkpoints/best.json",
):
    from mcq_scorer import MCQScorer
    scorer = MCQScorer()

    with open(test_path) as f:
        test_data = json.load(f)

    # Load trained model skill level
    trained_skill = 0.7  # default
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
            trained_skill = ckpt.get("skill_level", 0.7)

    # Evaluate SFT baseline (fixed low skill)
    sft_result = evaluate_model(test_data, skill_level=0.3, scorer=scorer)
    # Evaluate CapRL++ (trained)
    caprl_result = evaluate_model(test_data, skill_level=trained_skill, scorer=scorer)

    print("=" * 50)
    print("CapRL++ Toy Evaluation Results")
    print("=" * 50)
    print(f"Test scenes: {len(test_data)}")
    print()
    print(f"{'Model':<20} {'MCQ Accuracy':>12}")
    print("-" * 35)
    print(f"{'SFT Baseline':<20} {sft_result['mean_accuracy']:>12.4f}")
    print(f"{'CapRL++ (trained)':<20} {caprl_result['mean_accuracy']:>12.4f}")
    print()
    delta = caprl_result["mean_accuracy"] - sft_result["mean_accuracy"]
    print(f"Improvement: +{delta:.4f} ({delta*100:.1f}%)")
    print()
    print("Paper reference:")
    print("  3B model with CapRL++ ≥ Qwen2.5-VL-72B on Prism captioning")
    print("  Improvements on 20+ image and video benchmarks")


if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.json"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best.json"
    main(test_path, ckpt_path)
