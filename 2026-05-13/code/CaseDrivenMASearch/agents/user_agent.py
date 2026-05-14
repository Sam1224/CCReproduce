"""
User Agent — discovers bad cases through multi-turn conversational search simulation.

From paper §3.1: The User Agent simulates user behaviour in search interactions,
generating queries and identifying cases where retrieved products fail to satisfy
user intent (i.e., bad cases).
"""

import json
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchInteraction:
    query: str
    product_title: str
    product_desc: str
    predicted_label: str
    gold_label: Optional[str] = None
    is_bad_case: bool = False
    bad_case_reason: str = ""


class UserAgent:
    """
    Simulates user search interactions and identifies bad cases.

    Paper description:
        "The User Agent identifies bad cases through conversational interaction,
         mimicking user feedback signals such as query reformulation or explicit dissatisfaction."

    In production (paper):
        - Interfaces with live search logs
        - Uses user click/skip/reformulation signals
        - Operates in multi-turn dialogue mode

    In this toy reproduction:
        - Simulates user intent via rule-based heuristics
        - Generates synthetic bad cases by identifying label mismatches
    """

    RELEVANCE_LABELS = ["exact", "substitute", "complement", "irrelevant"]

    def __init__(self, relevance_threshold: str = "substitute"):
        """
        Args:
            relevance_threshold: Minimum acceptable relevance level.
                Labels below this are considered bad cases.
        """
        self.threshold_idx = self.RELEVANCE_LABELS.index(relevance_threshold)

    def _simulate_predicted_label(self, query: str, product_title: str) -> str:
        """Simulate a model's relevance prediction (toy: keyword overlap heuristic)."""
        query_terms = set(query.lower().split())
        title_terms = set(product_title.lower().split())
        overlap = len(query_terms & title_terms)

        if overlap >= 3:
            return "exact"
        elif overlap >= 2:
            return "substitute"
        elif overlap >= 1:
            return "complement"
        else:
            return "irrelevant"

    def _generate_bad_case_reason(
        self, query: str, predicted: str, gold: str
    ) -> str:
        if self.RELEVANCE_LABELS.index(predicted) < self.RELEVANCE_LABELS.index(gold):
            return f"Under-retrieval: model predicted '{predicted}' but gold is '{gold}'"
        else:
            return f"Over-retrieval: model predicted '{predicted}' but gold is '{gold}'"

    def find_bad_cases(
        self, pairs: List[dict], gold_labels: Optional[List[str]] = None
    ) -> List[SearchInteraction]:
        """
        Identify bad cases from query-product pairs.

        Args:
            pairs: List of {query, product_title, product_desc, label?} dicts
            gold_labels: Optional gold labels (if not in pairs)

        Returns:
            List of SearchInteraction, with is_bad_case=True for flagged items
        """
        interactions = []
        for i, pair in enumerate(pairs):
            gold = gold_labels[i] if gold_labels else pair.get("label")
            predicted = self._simulate_predicted_label(
                pair["query"], pair["product_title"]
            )
            is_bad = (
                gold is not None
                and self.RELEVANCE_LABELS.index(predicted)
                != self.RELEVANCE_LABELS.index(gold)
            )
            reason = ""
            if is_bad:
                reason = self._generate_bad_case_reason(pair["query"], predicted, gold)

            interactions.append(
                SearchInteraction(
                    query=pair["query"],
                    product_title=pair["product_title"],
                    product_desc=pair["product_desc"],
                    predicted_label=predicted,
                    gold_label=gold,
                    is_bad_case=is_bad,
                    bad_case_reason=reason,
                )
            )

        bad_cases = [x for x in interactions if x.is_bad_case]
        print(
            f"[UserAgent] Found {len(bad_cases)} bad cases out of {len(interactions)} pairs."
        )
        return interactions

    def report(self, interactions: List[SearchInteraction]) -> dict:
        bad = [x for x in interactions if x.is_bad_case]
        return {
            "total": len(interactions),
            "bad_cases": len(bad),
            "bad_case_rate": len(bad) / max(len(interactions), 1),
            "examples": [
                {
                    "query": b.query,
                    "product": b.product_title,
                    "predicted": b.predicted_label,
                    "gold": b.gold_label,
                    "reason": b.bad_case_reason,
                }
                for b in bad[:3]
            ],
        }
