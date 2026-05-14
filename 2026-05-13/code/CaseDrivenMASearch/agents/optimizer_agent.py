"""
Optimizer Agent — root-cause analysis and automated bad-case resolution.

From paper §3.3:
    "The Optimizer Agent autonomously analyzes bad cases and determines the root cause:
     (1) Policy Gap — the relevance policy specification is incomplete or ambiguous
     (2) Data Gap — the training data lacks coverage for this pattern
     (3) Model Gap — the model has sufficient data but fails to generalize

    Based on the diagnosis, the Optimizer triggers the appropriate resolution:
     - Policy Gap → update policy document / annotation guidelines
     - Data Gap → generate synthetic training samples covering the gap
     - Model Gap → adjust model architecture or hyperparameters"
"""

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RootCause(Enum):
    POLICY_GAP = "policy_gap"
    DATA_GAP = "data_gap"
    MODEL_GAP = "model_gap"


@dataclass
class Resolution:
    root_cause: RootCause
    diagnosis: str
    action: str
    synthetic_data: List[dict] = field(default_factory=list)
    policy_update: str = ""


@dataclass
class BadCase:
    query: str
    product_title: str
    product_desc: str
    predicted_label: str
    gold_label: Optional[str]
    bad_case_reason: str


class OptimizerAgent:
    """
    Diagnoses bad cases and triggers automated resolution.

    Root-cause classification heuristic (toy reproduction):
        - Policy Gap: if the predicted and gold labels are adjacent in the hierarchy
          (boundary ambiguity — likely a policy definition issue)
        - Data Gap: if the query pattern appears rarely in training data
        - Model Gap: otherwise (sufficient training coverage, model failure)
    """

    LABEL_ORDER = ["exact", "substitute", "complement", "irrelevant"]

    def __init__(self, training_pairs: Optional[List[dict]] = None):
        self.training_pairs = training_pairs or []
        # Build query term frequency index
        self._build_index()

    def _build_index(self):
        self.query_term_freq: dict = {}
        for pair in self.training_pairs:
            for term in pair.get("query", "").lower().split():
                self.query_term_freq[term] = self.query_term_freq.get(term, 0) + 1

    def _is_adjacent(self, label_a: str, label_b: str) -> bool:
        if label_a not in self.LABEL_ORDER or label_b not in self.LABEL_ORDER:
            return False
        idx_a = self.LABEL_ORDER.index(label_a)
        idx_b = self.LABEL_ORDER.index(label_b)
        return abs(idx_a - idx_b) == 1

    def _query_coverage(self, query: str) -> float:
        terms = query.lower().split()
        covered = sum(1 for t in terms if self.query_term_freq.get(t, 0) > 1)
        return covered / max(len(terms), 1)

    def _diagnose(self, bad_case: BadCase) -> RootCause:
        """Determine root cause of bad case (paper §3.3)."""
        if bad_case.gold_label and self._is_adjacent(
            bad_case.predicted_label, bad_case.gold_label
        ):
            return RootCause.POLICY_GAP

        coverage = self._query_coverage(bad_case.query)
        if coverage < 0.5:
            return RootCause.DATA_GAP

        return RootCause.MODEL_GAP

    def _generate_synthetic_samples(self, bad_case: BadCase, n: int = 3) -> List[dict]:
        """Generate synthetic training samples to cover a data gap."""
        # In production: LLM generates semantically similar query-product pairs
        # with correct labels. Here: create variants via simple augmentation.
        base_query = bad_case.query
        base_title = bad_case.product_title
        gold = bad_case.gold_label or "exact"

        variants = []
        query_words = base_query.split()
        for i in range(n):
            # Toy augmentation: shuffle/drop a word
            aug_query = " ".join(random.sample(query_words, max(len(query_words) - i % 2, 1)))
            variants.append({
                "query": aug_query,
                "product_title": base_title,
                "product_desc": bad_case.product_desc,
                "label": gold,
                "source": "synthetic_optimizer",
            })
        return variants

    def _policy_update_message(self, bad_case: BadCase) -> str:
        return (
            f"[Policy Update] Boundary case detected between "
            f"'{bad_case.predicted_label}' and '{bad_case.gold_label}' "
            f"for query='{bad_case.query}'. "
            f"Recommend clarifying policy: when product is a partial match "
            f"(only {len(set(bad_case.query.split()) & set(bad_case.product_title.split()))} "
            f"common terms), prefer '{bad_case.gold_label}' label. "
            f"Update annotation guidelines section 2.3."
        )

    def resolve(self, bad_case: BadCase) -> Resolution:
        """Diagnose and resolve a single bad case."""
        cause = self._diagnose(bad_case)

        if cause == RootCause.POLICY_GAP:
            policy_msg = self._policy_update_message(bad_case)
            return Resolution(
                root_cause=cause,
                diagnosis=(
                    f"Label boundary ambiguity: predicted={bad_case.predicted_label}, "
                    f"gold={bad_case.gold_label}. Adjacent labels indicate policy ambiguity."
                ),
                action="Update annotation policy guidelines",
                policy_update=policy_msg,
            )
        elif cause == RootCause.DATA_GAP:
            synthetic = self._generate_synthetic_samples(bad_case)
            return Resolution(
                root_cause=cause,
                diagnosis=(
                    f"Low training coverage for query '{bad_case.query}' "
                    f"(coverage={self._query_coverage(bad_case.query):.2f}). "
                    f"Insufficient training examples for this pattern."
                ),
                action="Generate synthetic training samples",
                synthetic_data=synthetic,
            )
        else:
            return Resolution(
                root_cause=cause,
                diagnosis=(
                    f"Model prediction error despite adequate training coverage. "
                    f"Predicted={bad_case.predicted_label}, gold={bad_case.gold_label}. "
                    f"Likely requires model-level intervention."
                ),
                action="Flag for hyperparameter tuning / fine-tuning round",
            )

    def resolve_batch(self, bad_cases: List[BadCase]) -> List[Resolution]:
        resolutions = [self.resolve(bc) for bc in bad_cases]
        cause_counts = {}
        for r in resolutions:
            cause_counts[r.root_cause.value] = cause_counts.get(r.root_cause.value, 0) + 1
        print(f"[OptimizerAgent] Resolved {len(resolutions)} bad cases: {cause_counts}")
        return resolutions

    def collect_synthetic_data(self, resolutions: List[Resolution]) -> List[dict]:
        """Collect all synthetic samples from data-gap resolutions."""
        synthetic = []
        for r in resolutions:
            if r.root_cause == RootCause.DATA_GAP:
                synthetic.extend(r.synthetic_data)
        return synthetic
