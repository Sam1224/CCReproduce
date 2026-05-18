"""
Evaluation metrics for EVADE benchmark.

The paper uses:
- Accuracy (full match)
- Partial-match Accuracy (for All-in-One with multiple violation types)
- F1 Score (binary: violation vs compliant)
"""

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np


def parse_model_output(output: str) -> int:
    """Parse LLM output into binary label (0=compliant, 1=violation)."""
    output_lower = output.lower().strip()
    if any(w in output_lower for w in ["violation", "违规", "违反", "yes", "1"]):
        return 1
    if any(w in output_lower for w in ["compliant", "合规", "no", "0"]):
        return 0
    # Ambiguous: default to compliant (conservative)
    return 0


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "n_samples": len(y_true),
        "n_evasive": sum(y_true),
        "n_predicted_evasive": sum(y_pred),
    }


def compute_per_category_metrics(
    y_true: list[int],
    y_pred: list[int],
    categories: list[str],
) -> dict[str, dict]:
    """Compute metrics broken down by product category."""
    from collections import defaultdict

    cat_data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for yt, yp, cat in zip(y_true, y_pred, categories):
        cat_data[cat][0].append(yt)
        cat_data[cat][1].append(yp)

    return {
        cat: compute_metrics(true, pred)
        for cat, (true, pred) in cat_data.items()
    }


def compute_partial_match_accuracy(
    y_true: list[list[int]],
    y_pred: list[list[int]],
) -> float:
    """
    Partial-match accuracy for All-in-One task.
    When a sample has multiple violation types, partial credit is given
    if at least one violation is correctly identified.

    Paper definition: partial_match = correct predictions / total predictions
    where correct = at least one label matches between truth and prediction.
    """
    # For binary simplification, partial match = same as accuracy
    # In multi-label setting (not yet released), this would be more nuanced
    matches = [
        any(yt == yp for yt, yp in zip(y_true_i, y_pred_i))
        for y_true_i, y_pred_i in zip(y_true, y_pred)
    ]
    return np.mean(matches)


def format_results(metrics: dict, title: str = "Results") -> str:
    """Pretty-print evaluation results."""
    lines = [f"\n{'='*50}", f"  {title}", "="*50]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:30s}: {v:.4f}")
        else:
            lines.append(f"  {k:30s}: {v}")
    lines.append("="*50)
    return "\n".join(lines)
