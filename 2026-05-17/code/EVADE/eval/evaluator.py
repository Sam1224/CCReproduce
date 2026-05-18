"""
Core evaluation framework for EVADE benchmark.
Supports Single-Violation and All-in-One tasks.
Implements the evaluation pipeline from the paper.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from eval.prompts import (
    build_single_violation_prompt,
    build_all_in_one_prompt,
    SYSTEM_PROMPT,
    POLICY_RULES,
)
from eval.metrics import parse_model_output, compute_metrics, compute_per_category_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    model: str = "gpt-4o"
    task_type: str = "single_violation"  # "single_violation" or "all_in_one"
    modality: str = "text"  # "text", "image", "multimodal"
    language: str = "zh"
    batch_size: int = 8
    max_tokens: int = 16
    temperature: float = 0.0
    output_path: Optional[str] = None


@dataclass
class EvalResult:
    sample_id: str
    category: str
    text: str
    true_label: int
    predicted_label: int
    raw_output: str
    prompt: str


class EVADEEvaluator:
    """
    Evaluator for EVADE benchmark.

    Implements two tasks from the paper:
    1. Single-Violation: Fine-grained detection with individual policy rules
    2. All-in-One: Long-context detection with unified policy instructions

    Paper finding: All-in-One setting significantly narrows the gap between
    partial and full-match accuracy, suggesting clearer rule definitions
    improve alignment between human and model judgment.
    """

    def __init__(self, config: EvalConfig, model_interface=None):
        self.config = config
        self.model = model_interface
        self.results: list[EvalResult] = []

    def _build_prompt(self, sample: dict) -> str:
        """Build task-specific prompt for a sample."""
        text = sample.get("text", "")
        category = sample.get("category", "health_supplements")
        rules = POLICY_RULES.get(category, [])

        if self.config.task_type == "single_violation":
            # For single violation: test against each rule separately
            # (simplification: use a random rule from the category)
            rule = rules[0] if rules else None
            return build_single_violation_prompt(
                text, category, rule, self.config.language
            )
        else:  # all_in_one
            return build_all_in_one_prompt(
                text, category, self.config.language
            )

    def evaluate(self, dataset) -> dict:
        """
        Run evaluation on the dataset.
        Returns aggregated metrics.
        """
        self.results = []
        y_true, y_pred, categories = [], [], []

        for sample in tqdm(dataset, desc=f"Evaluating ({self.config.task_type})"):
            prompt = self._build_prompt(sample)

            if self.model is not None:
                raw_output = self.model.predict(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            else:
                # Dummy model for testing
                raw_output = "合规" if sample.get("label", 0) == 0 else "违规"

            pred = parse_model_output(raw_output)
            true = sample.get("label", 0)
            cat = sample.get("category", "unknown")

            self.results.append(EvalResult(
                sample_id=sample.get("sample_id", ""),
                category=cat,
                text=sample.get("text", ""),
                true_label=true,
                predicted_label=pred,
                raw_output=raw_output,
                prompt=prompt,
            ))

            y_true.append(true)
            y_pred.append(pred)
            categories.append(cat)

        metrics = compute_metrics(y_true, y_pred)
        per_cat = compute_per_category_metrics(y_true, y_pred, categories)
        metrics["per_category"] = per_cat

        if self.config.output_path:
            self._save_results(metrics)

        return metrics

    def _save_results(self, metrics: dict) -> None:
        """Save evaluation results to JSON."""
        output = {
            "config": {
                "model": self.config.model,
                "task_type": self.config.task_type,
                "modality": self.config.modality,
            },
            "metrics": {k: v for k, v in metrics.items() if k != "per_category"},
            "per_category": metrics.get("per_category", {}),
            "predictions": [
                {
                    "sample_id": r.sample_id,
                    "category": r.category,
                    "true_label": r.true_label,
                    "predicted_label": r.predicted_label,
                    "raw_output": r.raw_output,
                }
                for r in self.results
            ],
        }
        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {self.config.output_path}")
