"""
Annotator Agent — multi-path chain-of-thought relevance labeling.

From paper §3.2:
    "The Annotator Agent produces relevance labels through multi-path reasoning,
     achieving 2.4% higher labeling precision than human annotators at 75.4% lower cost.
     The labeled data is used to train a Generative Relevance Model (GRM)."

Multi-path reasoning formula (paper, §3.2):
    For a query q and product p, the Annotator Agent generates K reasoning paths:
        {r_1, r_2, ..., r_K}
    Each path independently predicts a label l_k ∈ {exact, substitute, complement, irrelevant}.
    Final label: majority vote with confidence weighting:
        L* = argmax_{l} Σ_k w_k * 1[l_k == l]
    where w_k ∝ exp(confidence_k / T)  (temperature-scaled confidence)
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


LABEL_SPACE = ["exact", "substitute", "complement", "irrelevant"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABEL_SPACE)}


@dataclass
class ReasoningPath:
    path_id: int
    reasoning: str
    predicted_label: str
    confidence: float


@dataclass
class AnnotationResult:
    query: str
    product_title: str
    product_desc: str
    final_label: str
    confidence: float
    paths: List[ReasoningPath] = field(default_factory=list)
    rationale: str = ""


class AnnotatorAgent:
    """
    Multi-path chain-of-thought annotator for query-product relevance.

    In production (paper):
        - Uses a large LLM (e.g., GPT-4 or internal model) to generate K reasoning paths
        - Each path: [think step-by-step] → label
        - Aggregates with confidence-weighted majority vote

    In this toy reproduction:
        - Implements rule-based multi-path reasoning (3 paths per pair)
        - Path 1: semantic similarity heuristic
        - Path 2: category overlap heuristic
        - Path 3: specificity heuristic
        - Aggregates via temperature-scaled softmax vote
    """

    def __init__(self, num_paths: int = 3, temperature: float = 1.0):
        self.num_paths = num_paths
        self.T = temperature

    def _path_semantic(self, query: str, title: str, desc: str) -> Tuple[str, float, str]:
        """Path 1: semantic keyword overlap reasoning."""
        q_terms = set(query.lower().split())
        t_terms = set(title.lower().split())
        d_terms = set(desc.lower().split())
        overlap_title = len(q_terms & t_terms) / max(len(q_terms), 1)
        overlap_desc = len(q_terms & d_terms) / max(len(q_terms), 1)
        score = 0.7 * overlap_title + 0.3 * overlap_desc

        if score >= 0.5:
            label, conf = "exact", min(0.9, score + 0.2)
        elif score >= 0.3:
            label, conf = "substitute", 0.7
        elif score >= 0.1:
            label, conf = "complement", 0.6
        else:
            label, conf = "irrelevant", 0.85

        reasoning = (
            f"Semantic path: query-title overlap={overlap_title:.2f}, "
            f"query-desc overlap={overlap_desc:.2f}, score={score:.2f} → {label}"
        )
        return label, conf, reasoning

    def _path_category(self, query: str, title: str, desc: str) -> Tuple[str, float, str]:
        """Path 2: product category alignment reasoning."""
        # Toy category signals
        category_keywords = {
            "electronics": {"bluetooth", "wireless", "headphones", "laptop", "usb", "cable"},
            "footwear": {"shoes", "sneakers", "running", "athletic"},
            "kitchen": {"coffee", "maker", "mug", "ceramic"},
            "health": {"vitamin", "supplement", "mg"},
            "furniture": {"stand", "ergonomic", "adjustable"},
            "appliance": {"purifier", "hepa", "filter"},
            "accessories": {"mouse", "gaming", "rgb"},
        }

        def detect_category(text: str) -> Optional[str]:
            text_lower = text.lower()
            for cat, keywords in category_keywords.items():
                if any(k in text_lower for k in keywords):
                    return cat
            return None

        query_cat = detect_category(query)
        product_cat = detect_category(title + " " + desc)

        if query_cat is not None and query_cat == product_cat:
            label, conf = "exact", 0.8
        elif query_cat is not None and product_cat is not None:
            label, conf = "substitute", 0.65
        elif query_cat is None or product_cat is None:
            label, conf = "complement", 0.5
        else:
            label, conf = "irrelevant", 0.75

        reasoning = (
            f"Category path: query_cat={query_cat}, product_cat={product_cat} → {label}"
        )
        return label, conf, reasoning

    def _path_specificity(self, query: str, title: str, desc: str) -> Tuple[str, float, str]:
        """Path 3: specificity / attribute matching reasoning."""
        # Extract numeric attributes from query
        import re
        query_numbers = set(re.findall(r'\d+', query))
        title_numbers = set(re.findall(r'\d+', title + " " + desc))
        number_match = len(query_numbers & title_numbers) / max(len(query_numbers), 1) if query_numbers else 0.5

        q_terms = set(query.lower().split())
        t_terms = set(title.lower().split())
        word_match = len(q_terms & t_terms) / max(len(q_terms), 1)

        combined = 0.4 * number_match + 0.6 * word_match
        if combined >= 0.5:
            label, conf = "exact", min(0.95, combined + 0.3)
        elif combined >= 0.25:
            label, conf = "substitute", 0.65
        elif combined >= 0.1:
            label, conf = "complement", 0.55
        else:
            label, conf = "irrelevant", 0.8

        reasoning = (
            f"Specificity path: number_match={number_match:.2f}, "
            f"word_match={word_match:.2f}, combined={combined:.2f} → {label}"
        )
        return label, conf, reasoning

    def _aggregate_paths(self, paths: List[ReasoningPath]) -> Tuple[str, float]:
        """
        Temperature-scaled confidence-weighted majority vote.

        L* = argmax_{l} Σ_k softmax(confidence_k / T)[k] * 1[l_k == l]
        """
        label_scores = {l: 0.0 for l in LABEL_SPACE}
        confs = torch.tensor([p.confidence for p in paths])
        weights = F.softmax(confs / self.T, dim=0)

        for path, w in zip(paths, weights.tolist()):
            label_scores[path.predicted_label] += w

        final_label = max(label_scores, key=label_scores.__getitem__)
        final_conf = label_scores[final_label]
        return final_label, final_conf

    def annotate(self, pair: dict) -> AnnotationResult:
        """Annotate a single query-product pair with multi-path reasoning."""
        q, t, d = pair["query"], pair["product_title"], pair["product_desc"]

        path_fns = [self._path_semantic, self._path_category, self._path_specificity]
        paths = []
        for i, fn in enumerate(path_fns[:self.num_paths]):
            label, conf, reasoning = fn(q, t, d)
            paths.append(ReasoningPath(i, reasoning, label, conf))

        final_label, final_conf = self._aggregate_paths(paths)
        rationale = " | ".join(p.reasoning for p in paths)

        return AnnotationResult(
            query=q,
            product_title=t,
            product_desc=d,
            final_label=final_label,
            confidence=final_conf,
            paths=paths,
            rationale=rationale,
        )

    def annotate_batch(self, pairs: List[dict]) -> List[AnnotationResult]:
        results = [self.annotate(p) for p in pairs]
        print(f"[AnnotatorAgent] Annotated {len(results)} pairs.")
        return results

    def compute_precision(
        self, results: List[AnnotationResult], gold_labels: List[str]
    ) -> float:
        correct = sum(
            r.final_label == g for r, g in zip(results, gold_labels) if g is not None
        )
        total = sum(1 for g in gold_labels if g is not None)
        return correct / max(total, 1)
