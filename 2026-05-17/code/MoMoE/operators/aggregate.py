"""
Aggregate Operator for MoMoE.

Combines predictions from multiple experts using:
1. Weighted dot product (uses allocation weights from Allocator)
2. Simple majority vote

Paper: "The Aggregate operator combines the predictions of the
selected top-K experts using either a weighted dot product or
a simple majority vote strategy."
"""

from dataclasses import dataclass
from models.expert import ExpertPrediction
from models.allocator import AllocationResult


@dataclass
class AggregateResult:
    label: int          # Final binary prediction
    confidence: float   # Aggregated confidence score
    strategy: str       # "weighted" or "majority_vote"


def aggregate_weighted(
    expert_predictions: dict[str, list[ExpertPrediction]],
    allocation: AllocationResult,
    sample_idx: int = 0,
) -> AggregateResult:
    """
    Weighted dot product aggregation.

    Formula (from paper):
    score = Σ_k w_k * p_k(violation)

    where w_k is the allocation weight for expert k,
    p_k(violation) is expert k's predicted violation probability.

    Final label = 1 if score >= 0.5 else 0
    """
    if not expert_predictions:
        return AggregateResult(label=0, confidence=0.5, strategy="weighted")

    weighted_score = 0.0
    total_weight = 0.0

    for expert_id, weight in zip(allocation.top_k_indices, allocation.top_k_weights):
        expert_label = allocation.expert_labels[allocation.top_k_indices.index(expert_id)]
        if expert_label in expert_predictions:
            preds = expert_predictions[expert_label]
            if sample_idx < len(preds):
                pred = preds[sample_idx]
                weighted_score += weight * pred.confidence
                total_weight += weight

    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.5

    return AggregateResult(
        label=1 if final_score >= 0.5 else 0,
        confidence=final_score,
        strategy="weighted",
    )


def aggregate_majority_vote(
    expert_predictions: dict[str, list[ExpertPrediction]],
    sample_idx: int = 0,
) -> AggregateResult:
    """
    Simple majority vote aggregation.

    Each active expert gets one vote; majority determines the final label.
    Ties broken in favor of 'not violation' (conservative).
    """
    votes = []
    for expert_id, preds in expert_predictions.items():
        if sample_idx < len(preds):
            votes.append(preds[sample_idx].label)

    if not votes:
        return AggregateResult(label=0, confidence=0.5, strategy="majority_vote")

    violation_count = sum(votes)
    compliant_count = len(votes) - violation_count
    label = 1 if violation_count > compliant_count else 0
    confidence = violation_count / len(votes)

    return AggregateResult(
        label=label,
        confidence=confidence,
        strategy="majority_vote",
    )


def aggregate_batch(
    expert_predictions: dict[str, list[ExpertPrediction]],
    allocations: list[AllocationResult],
    strategy: str = "weighted",
) -> list[AggregateResult]:
    """Aggregate predictions for a full batch."""
    n = len(allocations)
    results = []

    for i in range(n):
        if strategy == "weighted":
            result = aggregate_weighted(expert_predictions, allocations[i], sample_idx=i)
        else:
            result = aggregate_majority_vote(expert_predictions, sample_idx=i)
        results.append(result)

    return results
