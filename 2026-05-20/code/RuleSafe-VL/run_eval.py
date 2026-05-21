"""
RuleSafe-VL Evaluation Entry Point
=====================================
Run:
  python run_eval.py --model random --cases data/toy_cases.jsonl
  python run_eval.py --model llava-1.6 --cases data/toy_cases.jsonl --output results/llava.json
  python run_eval.py --model gpt-4o --cases data/real_cases.jsonl --output results/gpt4o.json

Arguments:
  --model    : model backend (random | llava-1.6 | gpt-4o | gpt-5.2)
  --cases    : path to JSONL case file
  --output   : (optional) path to save JSON results
  --verbose  : (flag) print per-case results
  --gen-toy  : (flag) generate toy dataset and exit
"""

import argparse
import sys
from pathlib import Path

# Module path setup
sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent / "models"))


def main():
    parser = argparse.ArgumentParser(description="RuleSafe-VL Evaluator")
    parser.add_argument("--model", default="random",
                        help="Model backend: random | llava-1.6 | gpt-4o")
    parser.add_argument("--cases", default="data/toy_cases.jsonl",
                        help="Path to benchmark cases JSONL")
    parser.add_argument("--output", default=None,
                        help="Output path for results JSON")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--gen-toy", action="store_true",
                        help="Generate toy dataset and exit")
    args = parser.parse_args()

    if args.gen_toy:
        from toy_dataset import generate_toy_cases
        generate_toy_cases(output_path=args.cases)
        print("Toy dataset generated. Re-run without --gen-toy to evaluate.")
        return

    # Import after path setup
    from vlm_wrapper import get_model_wrapper
    from rule_graph import RuleGraph
    from evaluator import RuleSafeVLEvaluator

    print(f"Loading model: {args.model}")
    model = get_model_wrapper(args.model)
    rule_graph = RuleGraph()
    print(rule_graph.summary())

    evaluator = RuleSafeVLEvaluator(
        model_wrapper=model,
        rule_graph=rule_graph,
        verbose=args.verbose,
    )
    metrics = evaluator.run(cases_path=args.cases, output_path=args.output)
    return metrics


if __name__ == "__main__":
    main()
