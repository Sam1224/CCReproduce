"""
RuleSafe-VL Benchmark Runner

Usage:
    python main.py --model_type openai --model_name gpt-4o --rules_path data/rules.json --cases_path data/sample_cases.json
    python main.py --model_type huggingface --model_name Qwen/Qwen2-VL-7B-Instruct --rules_path data/rules.json --cases_path data/sample_cases.json
    python main.py --demo   # Run demo with rule graph inspection only (no VLM)

Based on: arXiv:2605.07760
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.rule_graph import RuleGraph, PolicyFamily
from models.vlm_evaluator import build_evaluator
from eval.benchmark import load_cases, run_benchmark


def demo_rule_graph(rules_path: str):
    """Demo: load rule graph and inspect its structure without running VLM."""
    graph = RuleGraph.from_json(rules_path)
    print(f"\n{'='*60}")
    print("RuleSafe-VL Rule Graph Summary")
    print(f"{'='*60}")
    print(f"  Total rules: {len(graph.rules)}")
    print(f"  Total relations: {len(graph.relations)}")

    for fam in PolicyFamily:
        fam_rules = [r for r in graph.rules.values() if r.policy_family == fam]
        print(f"\n  [{fam.value}]  {len(fam_rules)} rules")
        for rule in fam_rules:
            print(f"    {rule.rule_id}: {rule.description[:70]}...")

    print(f"\n{'='*60}")
    print("Rule Relations")
    print(f"{'='*60}")
    from collections import Counter
    from models.rule_graph import RelationType
    rel_types = Counter(rel.relation_type.value for rel in graph.relations.values())
    for rtype, count in rel_types.most_common():
        print(f"  {rtype:15s}: {count}")

    print(f"\n{'='*60}")
    print("Sample Policy Context (hate_speech family)")
    print(f"{'='*60}")
    print(graph.to_prompt_context(policy_family=PolicyFamily.HATE_SPEECH))


def demo_oracle_evaluation(rules_path: str, cases_path: str):
    """
    Demo: evaluate with oracle (perfect) predictions to verify metrics computation.
    Useful for validating the evaluation pipeline without a real VLM.
    """
    from models.vlm_evaluator import BaseVLMEvaluator, VLMResponse
    from typing import List, Optional

    class OracleEvaluator(BaseVLMEvaluator):
        """Always returns ground truth — useful for pipeline validation."""

        def __init__(self, cases):
            self.gt_map = {c["case_id"]: c for c in cases}

        def evaluate_case(self, image_path, text_content, policy_context, rule_ids):
            # For oracle, we'd need case_id — hack: match by text content
            for c in self.gt_map.values():
                if c["text_content"] == text_content:
                    return VLMResponse(
                        raw_text="[Oracle]",
                        activated_rule_ids=set(c["gt_activated_rules"]),
                        decision=c["gt_decision"],
                        sufficient_evidence=c["gt_sufficient_evidence"],
                        reasoning="Oracle prediction",
                    )
            return VLMResponse(
                raw_text="[Oracle miss]",
                activated_rule_ids=set(),
                decision="allowed",
                sufficient_evidence=True,
                reasoning="",
            )

    graph = RuleGraph.from_json(rules_path)
    cases = load_cases(cases_path)
    evaluator = OracleEvaluator(cases)

    print("\n[Oracle Evaluation — should show perfect scores]")
    run_benchmark(
        evaluator=evaluator,
        rule_graph=graph,
        cases=cases,
        model_name="Oracle",
        save_results=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="RuleSafe-VL Benchmark Evaluation (arXiv:2605.07760)"
    )
    parser.add_argument(
        "--model_type",
        choices=["openai", "huggingface"],
        default="openai",
        help="VLM backend: 'openai' or 'huggingface'",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-4o",
        help="Model name (e.g. 'gpt-4o', 'Qwen/Qwen2-VL-7B-Instruct')",
    )
    parser.add_argument(
        "--rules_path",
        default="data/rules.json",
        help="Path to rules JSON file",
    )
    parser.add_argument(
        "--cases_path",
        default="data/sample_cases.json",
        help="Path to test cases JSON file",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Directory containing case images (optional)",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory for saving evaluation results",
    )
    parser.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode: inspect rule graph + oracle evaluation",
    )
    parser.add_argument(
        "--generate_data",
        action="store_true",
        help="Generate toy data before running evaluation",
    )

    args = parser.parse_args()

    # Optionally generate toy data first
    if args.generate_data:
        print("Generating toy data...")
        import subprocess
        subprocess.run([sys.executable, "data/generate_toy_data.py", "--output", "data/"])

    if args.demo:
        demo_rule_graph(args.rules_path)
        demo_oracle_evaluation(args.rules_path, args.cases_path)
        return

    # Check data files exist
    if not Path(args.rules_path).exists():
        print(f"Rules file not found: {args.rules_path}")
        print("Run: python data/generate_toy_data.py --output data/")
        sys.exit(1)
    if not Path(args.cases_path).exists():
        print(f"Cases file not found: {args.cases_path}")
        print("Run: python data/generate_toy_data.py --output data/")
        sys.exit(1)

    # Load resources
    print(f"Loading rule graph from {args.rules_path}...")
    graph = RuleGraph.from_json(args.rules_path)
    print(f"  {len(graph.rules)} rules, {len(graph.relations)} relations loaded")

    print(f"Loading cases from {args.cases_path}...")
    cases = load_cases(args.cases_path)
    print(f"  {len(cases)} cases loaded")

    # Initialize VLM evaluator
    print(f"\nInitializing {args.model_type} evaluator: {args.model_name}...")
    kwargs = {}
    if args.model_type == "openai" and args.openai_api_key:
        kwargs["api_key"] = args.openai_api_key

    evaluator = build_evaluator(args.model_type, args.model_name, **kwargs)

    # Run benchmark
    results = run_benchmark(
        evaluator=evaluator,
        rule_graph=graph,
        cases=cases,
        image_dir=args.image_dir,
        model_name=args.model_name,
        save_results=True,
        output_dir=args.output_dir,
    )

    print(f"\nEvaluation complete. {len(results)} cases processed.")


if __name__ == "__main__":
    main()
