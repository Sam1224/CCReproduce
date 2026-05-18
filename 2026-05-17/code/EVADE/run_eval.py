"""
Main evaluation script for EVADE benchmark.

Usage:
    python run_eval.py --mode single_violation --model dummy
    python run_eval.py --mode all_in_one --model gpt-4o
    python run_eval.py --mode single_violation --model claude-opus-4-7 --data_path /path/to/data
"""

import argparse
import logging
import json
from pathlib import Path

from data.toy_data import generate_toy_dataset
from data.dataset import EVADEDataset
from eval.evaluator import EVADEEvaluator, EvalConfig
from eval.metrics import format_results
from models.model_interface import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="EVADE Benchmark Evaluation")
    parser.add_argument("--mode", choices=["single_violation", "all_in_one"], default="single_violation")
    parser.add_argument("--model", default="dummy", help="Model: dummy, gpt-4o, claude-opus-4-7, etc.")
    parser.add_argument("--modality", choices=["text", "image", "multimodal"], default="text")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--data_path", default=None, help="Path to EVADE dataset (uses toy data if not provided)")
    parser.add_argument("--n_toy", type=int, default=200, help="Number of toy samples to generate")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"EVADE Evaluation | task={args.mode} | model={args.model}")

    # Load data
    if args.data_path:
        logger.info(f"Loading dataset from {args.data_path}")
        dataset = EVADEDataset(args.data_path, task_type=args.mode, modality=args.modality)
    else:
        logger.info(f"Using toy dataset ({args.n_toy} samples)")
        toy_samples = generate_toy_dataset(n_samples=args.n_toy, task_type=args.mode)
        dataset = [
            {
                "sample_id": s.sample_id,
                "category": s.category,
                "text": s.text,
                "label": s.label,
                "task_type": s.task_type,
            }
            for s in toy_samples
        ]

    # Load model
    model = get_model(args.model)

    # Configure evaluation
    config = EvalConfig(
        model=args.model,
        task_type=args.mode,
        modality=args.modality,
        language=args.language,
        output_path=args.output,
    )

    # Run evaluation
    evaluator = EVADEEvaluator(config, model_interface=model)
    metrics = evaluator.evaluate(dataset)

    # Print results
    print(format_results(
        {k: v for k, v in metrics.items() if k != "per_category"},
        title=f"EVADE Results ({args.mode}, {args.model})"
    ))

    # Print per-category breakdown
    if "per_category" in metrics:
        print("\n--- Per-Category F1 ---")
        for cat, cat_metrics in metrics["per_category"].items():
            print(f"  {cat:25s}: F1={cat_metrics['f1']:.4f}, "
                  f"Acc={cat_metrics['accuracy']:.4f}, "
                  f"N={cat_metrics['n_samples']}")

    logger.info(f"Results saved to {args.output}")
    return metrics


if __name__ == "__main__":
    main()
