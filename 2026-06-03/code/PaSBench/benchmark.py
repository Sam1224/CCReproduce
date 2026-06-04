"""
PaSBench-Video Benchmark Runner

Usage:
    python benchmark.py --judge random --split all
    python benchmark.py --judge motion --split risk
    python benchmark.py --judge transformer --device cuda --max_samples 100

Paper: arXiv:2606.02443
"""

import argparse
import torch

from data.dataset import ToyPaSBenchDataset, PaSBenchVideoDataset
from models.mllm_judge import RandomJudge, SimpleMotionJudge, TransformerJudge
from evaluation.evaluator import PaSBenchEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="PaSBench-Video Benchmark")
    p.add_argument(
        "--judge",
        choices=["random", "motion", "transformer"],
        default="transformer",
        help="Which judge model to use",
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="Path to PaSBench-Video dataset. If None, uses toy data.",
    )
    p.add_argument(
        "--split",
        choices=["all", "risk", "no_risk"],
        default="all",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--motion_threshold", type=float, default=0.15)
    p.add_argument("--warn_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"PaSBench-Video Benchmark")
    print(f"  Judge: {args.judge}")
    print(f"  Split: {args.split}")
    print(f"  Device: {args.device}")
    print()

    # Load dataset
    if args.data_dir:
        print(f"Loading real dataset from {args.data_dir}")
        dataset = PaSBenchVideoDataset(data_dir=args.data_dir, split=args.split)
    else:
        print("Using toy dataset (no data_dir provided)")
        dataset = ToyPaSBenchDataset(split=args.split, seed=args.seed)
    print(f"Dataset size: {len(dataset)} samples")

    # Initialize judge
    if args.judge == "random":
        judge = RandomJudge(p_warn_per_frame=0.05, seed=args.seed)
    elif args.judge == "motion":
        judge = SimpleMotionJudge(motion_threshold=args.motion_threshold)
    else:  # transformer
        from models.mllm_judge import TemporalSafetyTransformer
        model = TemporalSafetyTransformer(
            img_size=224, patch_size=16, d_model=256, nhead=8, num_layers=4
        )
        judge = TransformerJudge(model=model, threshold=args.warn_threshold, device=args.device)
        print(f"TemporalSafetyTransformer parameters: "
              f"{sum(p.numel() for p in model.parameters()):,}")

    # Run evaluation
    evaluator = PaSBenchEvaluator(judge=judge, device=args.device, verbose=True)
    metrics = evaluator.evaluate_dataset(dataset, max_samples=args.max_samples)

    print(f"\nNote: Best published model (Seed-2.0-Pro) achieves <20.0% strict score.")
    print(f"      Most open-source models remain below 25% on this benchmark.")


if __name__ == "__main__":
    main()
