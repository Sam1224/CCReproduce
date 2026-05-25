"""
Main benchmark runner for RuleSafe-VL.

Orchestrates:
  1. Load rule graph from JSON
  2. Load test cases
  3. Run VLM evaluation on each case
  4. Compute and report metrics

Based on: arXiv:2605.07760
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from models.rule_graph import RuleGraph, PolicyFamily
from models.vlm_evaluator import BaseVLMEvaluator
from eval.metrics import CaseResult, evaluate_case, aggregate_metrics, print_metrics


def load_cases(cases_path: str) -> List[Dict]:
    with open(cases_path) as f:
        return json.load(f)


def run_benchmark(
    evaluator: BaseVLMEvaluator,
    rule_graph: RuleGraph,
    cases: List[Dict],
    image_dir: Optional[str] = None,
    model_name: str = "VLM",
    save_results: bool = True,
    output_dir: str = "results",
) -> List[CaseResult]:
    """
    Run the full RuleSafe-VL evaluation pipeline.

    Args:
        evaluator: Initialized VLM evaluator (OpenAI or HuggingFace)
        rule_graph: Loaded RuleGraph with rules and relations
        cases: List of test cases (from sample_cases.json)
        image_dir: Optional directory containing case images
        model_name: Name used in output files and reports
        save_results: Whether to save per-case results to JSON
        output_dir: Directory for saving results
    """
    results: List[CaseResult] = []

    for case in tqdm(cases, desc=f"Evaluating {model_name}"):
        case_id = case["case_id"]
        policy_family = case["policy_family"]
        text_content = case["text_content"]
        image_file = case.get("image_file")

        # Resolve image path
        image_path = None
        if image_file and image_dir:
            candidate = Path(image_dir) / image_file
            if candidate.exists():
                image_path = str(candidate)

        # Build policy context from the relevant policy family
        pf = PolicyFamily(policy_family)
        policy_context = rule_graph.to_prompt_context(policy_family=pf)

        # Get all rule IDs for this policy family
        rule_ids = [
            rid for rid, rule in rule_graph.rules.items()
            if rule.policy_family == pf
        ]

        # VLM inference
        try:
            vlm_response = evaluator.evaluate_case(
                image_path=image_path,
                text_content=text_content,
                policy_context=policy_context,
                rule_ids=rule_ids,
            )
        except Exception as e:
            print(f"  Error on case {case_id}: {e}")
            vlm_response = None

        # Build CaseResult
        gt_activated = set(case.get("gt_activated_rules", []))
        gt_relations = set(case.get("gt_applicable_relations", []))
        gt_sufficient = case.get("gt_sufficient_evidence", True)
        gt_decision = case.get("gt_decision", "allowed")

        if vlm_response is not None:
            case_result = CaseResult(
                case_id=case_id,
                policy_family=policy_family,
                gt_activated_rules=gt_activated,
                gt_applicable_relations=gt_relations,
                gt_sufficient_evidence=gt_sufficient,
                gt_decision=gt_decision,
                pred_activated_rules=vlm_response.activated_rule_ids,
                pred_applicable_relations=set(),  # Not currently extracted by parser
                pred_sufficient_evidence=vlm_response.sufficient_evidence,
                pred_decision=vlm_response.decision,
            )
        else:
            # Failure case: all predictions empty
            case_result = CaseResult(
                case_id=case_id,
                policy_family=policy_family,
                gt_activated_rules=gt_activated,
                gt_applicable_relations=gt_relations,
                gt_sufficient_evidence=gt_sufficient,
                gt_decision=gt_decision,
                pred_activated_rules=set(),
                pred_applicable_relations=set(),
                pred_sufficient_evidence=True,
                pred_decision="allowed",
            )

        evaluate_case(case_result)
        results.append(case_result)

    # Compute aggregate metrics
    metrics = aggregate_metrics(results)
    print_metrics(metrics, model_name=model_name)

    # Save results
    if save_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        results_path = Path(output_dir) / f"{safe_model_name}_results.json"
        results_dict = [
            {
                "case_id": r.case_id,
                "policy_family": r.policy_family,
                "rule_activation_f1": r.rule_activation_f1,
                "interaction_accuracy": r.interaction_accuracy,
                "sufficiency_correct": r.sufficiency_correct,
                "decision_correct": r.decision_correct,
                "gt_decision": r.gt_decision,
                "pred_decision": r.pred_decision,
            }
            for r in results
        ]
        with open(results_path, "w") as f:
            json.dump(
                {"model": model_name, "metrics": metrics.__dict__, "cases": results_dict},
                f,
                indent=2,
            )
        print(f"  Results saved to: {results_path}")

    return results
