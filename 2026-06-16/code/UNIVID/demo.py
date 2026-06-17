"""
Demo: Run the full UNIVID three-stage pipeline on toy data.

Outputs per-segment decisions with stage routing and latency breakdown.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.toy_dataset import ToyVideoDataset
from pipeline.three_stage_pipeline import build_default_pipeline


def main():
    device = "cpu"
    ds = ToyVideoDataset(num_samples=50, seed=99, split="demo")
    pipeline = build_default_pipeline(device=device)

    print("=== UNIVID Three-Stage Pipeline Demo ===")
    print(f"{'ID':<12} {'GT':<10} {'Pred':<10} {'Stage':<14} {'Risk':>6} {'Conf':>6} {'ms':>6}")
    print("-" * 70)

    correct = 0
    for seg in ds.samples[:10]:
        result = pipeline.moderate_segment(
            segment_id=seg.segment_id,
            text=seg.text,
            visual_feat=seg.visual_feat,
        )
        gt = "VIOLATION" if seg.label == 1 else "SAFE"
        pred = "VIOLATION" if result.is_violation else "SAFE"
        match = "✓" if (result.is_violation == seg.label) else "✗"
        correct += int(result.is_violation == seg.label)

        print(f"{seg.segment_id:<12} {gt:<10} {pred:<10} "
              f"{result.stage:<14} "
              f"{result.risk_score:>6.3f} "
              f"{result.confidence:>6.3f} "
              f"{result.latency_ms:>6.1f} {match}")

    print(f"\nAccuracy on 10 samples: {correct}/10")
    print("\nPaper results (production):")
    print("  Risk Filter:  >95% recall, filters ~60% of safe traffic")
    print("  UNIVID-Lite:  Standard categories, <5ms latency")
    print("  UNIVID-RAG:   +9% recall on novel categories")
    print("  Trend system: Detects new patterns within hours of emergence")


if __name__ == "__main__":
    main()
