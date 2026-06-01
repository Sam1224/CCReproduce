#!/usr/bin/env python
"""Generate toy livestream content moderation dataset."""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import generate_toy_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cases", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="data/toy_cases")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_toy_cases(args.num_cases, out_dir=args.out_dir, seed=args.seed)
