"""
RuleSafe-VL Main Evaluator
============================
Orchestrates the evaluation pipeline:
  1. Load benchmark cases
  2. Load rule graph
  3. For each case, query a VLM with the structured prompt
  4. Parse VLM response
  5. Compute all four task metrics
  6. Save results

Usage:
  python run_eval.py --model llava-1.6 --cases data/toy_cases.jsonl
"""

import sys
import os
import json
import time
from typing import List, Optional
from pathlib import Path

# Allow imports from sibling modules when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from case_format import BenchmarkCase, load_cases
from rule_graph import RuleGraph, PolicyFamily
from metrics import CaseResult, compute_all_metrics, print_metrics
from prompts import build_task_abc_prompt, parse_vlm_response


class RuleSafeVLEvaluator:
    def __init__(self, model_wrapper, rule_graph: Optional[RuleGraph] = None,
                 verbose: bool = True, sleep_between: float = 0.5):
        self.model = model_wrapper
        self.rule_graph = rule_graph or RuleGraph()
        self.verbose = verbose
        self.sleep_between = sleep_between

    def evaluate_case(self, case: BenchmarkCase) -> CaseResult:
        """Run single-case evaluation."""
        # Get rules for the relevant policy family (paper uses all-family context)
        # Simplified: use rules from the case's policy family + common cross-family rules
        family_map = {
            "F1": PolicyFamily.NUDITY_SEXUAL,
            "F2": PolicyFamily.DANGEROUS_HARMFUL,
            "F3": PolicyFamily.GRAPHIC_INJURY,
        }
        target_family = family_map.get(case.policy_family, PolicyFamily.NUDITY_SEXUAL)
        rules = self.rule_graph.get_applicable_rules(target_family)
        rules_dicts = [
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "conditions": r.conditions,
                "exceptions": r.exceptions,
            }
            for r in rules
        ]

        prompt = build_task_abc_prompt(
            rules=rules_dicts,
            text_context=case.text_context,
            platform_context=case.platform_context,
            image_description=f"[Image at {case.image_path}]" if case.image_path else None,
        )

        response_text = self.model.generate(
            prompt=prompt,
            image_path=case.image_path,
        )

        parsed = parse_vlm_response(response_text)

        if self.verbose:
            fallback = " [FALLBACK PARSE]" if parsed.get("_parse_fallback") else ""
            print(f"  Case {case.case_id}: predicted {parsed.get('moderation_outcome')} "
                  f"(gold: {case.moderation_outcome.value}){fallback}")

        return CaseResult(
            case_id=case.case_id,
            pred_applicable=parsed.get("applicable_rules", []),
            gold_applicable=case.applicable_rules,
            pred_violated=parsed.get("violated_rules", []),
            gold_violated=case.violated_rules,
            pred_outcome=parsed.get("moderation_outcome", "ALLOW"),
            gold_outcome=case.moderation_outcome.value,
        )

    def evaluate_dataset(self, cases: List[BenchmarkCase]) -> List[CaseResult]:
        results = []
        for i, case in enumerate(cases):
            if self.verbose:
                print(f"[{i+1}/{len(cases)}] Evaluating {case.case_id} ({case.policy_family})...")
            result = self.evaluate_case(case)
            results.append(result)
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)
        return results

    def run(self, cases_path: str, output_path: Optional[str] = None) -> dict:
        cases = load_cases(cases_path)
        print(f"Loaded {len(cases)} cases from {cases_path}")

        results = self.evaluate_dataset(cases)
        metrics = compute_all_metrics(results)
        print_metrics(metrics)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output = {
                "metrics": metrics,
                "per_case": [
                    {
                        "case_id": r.case_id,
                        "pred_applicable": r.pred_applicable,
                        "gold_applicable": r.gold_applicable,
                        "pred_violated": r.pred_violated,
                        "gold_violated": r.gold_violated,
                        "pred_outcome": r.pred_outcome,
                        "gold_outcome": r.gold_outcome,
                    }
                    for r in results
                ]
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {output_path}")

        return metrics
