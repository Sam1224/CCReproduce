"""
GLiGuard — Evaluation script
Multi-aspect safety classification evaluation.

Usage:
    python evaluate.py --checkpoint checkpoints/gliguard_best.pt
                       --schema prompt_safety response_safety violence
"""

import argparse
import logging
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from data import LLMSafetyDataset, collate_fn, ALL_SCHEMA_DIMS
from model import GLiGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    model: GLiGuard,
    loader: DataLoader,
    device: str = "cpu",
    schema_dims_to_eval: list = None,
) -> dict:
    """
    Evaluate GLiGuard on multi-aspect safety classification.
    Reports per-dimension F1, precision, recall, and latency.
    """
    model.eval()
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    total_samples, total_time = 0, 0.0

    with torch.no_grad():
        for batch in loader:
            t0 = time.time()
            preds = model.predict(
                batch["prompts"],
                batch["responses"],
                batch["active_schemas"],
            )
            t1 = time.time()
            total_time += (t1 - t0)
            total_samples += len(batch["prompts"])

            for b_idx, (pred_dict, label_dict, schema) in enumerate(
                zip(preds, batch["labels"], batch["active_schemas"])
            ):
                for dim in schema:
                    if dim in label_dict and dim in pred_dict:
                        all_preds[dim].append(pred_dict[dim])
                        all_labels[dim].append(label_dict[dim])

    # Per-dimension metrics
    dim_metrics = {}
    dims_to_report = schema_dims_to_eval or ["prompt_safety", "response_safety",
                                              "refusal_detection", "violence",
                                              "hate_speech", "role_play"]

    for dim in dims_to_report:
        if dim not in all_labels or len(all_labels[dim]) < 2:
            continue
        y_true = all_labels[dim]
        y_pred = all_preds[dim]
        dim_metrics[dim] = {
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "n": len(y_true),
        }

    # Overall metrics
    all_y_true = [v for vals in all_labels.values() for v in vals]
    all_y_pred = [v for vals in all_preds.values() for v in vals]
    overall_f1 = f1_score(all_y_true, all_y_pred, average="binary", zero_division=0) \
        if all_y_true else 0.0

    throughput = total_samples / total_time if total_time > 0 else 0.0
    avg_latency_ms = (total_time / total_samples * 1000) if total_samples > 0 else 0.0

    return {
        "overall_f1": overall_f1,
        "dim_metrics": dim_metrics,
        "throughput_samples_per_sec": throughput,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": total_samples,
    }


def parse_args():
    p = argparse.ArgumentParser(description="GLiGuard evaluation")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--schema", nargs="*",
                   default=["prompt_safety", "response_safety", "refusal_detection",
                            "violence", "hate_speech", "role_play"])
    return p.parse_args()


def main():
    args = parse_args()

    ds = LLMSafetyDataset(split="test", num_samples=args.num_samples, seed=12345)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GLiGuard(schema_dims=ALL_SCHEMA_DIMS, device=args.device).to(args.device)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state)

    results = evaluate(model, loader, device=args.device, schema_dims_to_eval=args.schema)

    print("\n" + "="*65)
    print("  GLiGuard — Multi-Aspect Safety Evaluation")
    print("="*65)
    print(f"  Overall F1:              {results['overall_f1']:.4f}")
    print(f"  Throughput:              {results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Avg Latency:             {results['avg_latency_ms']:.2f} ms/sample")
    print()
    print(f"  {'Dimension':<30} {'F1':>6} {'Prec':>6} {'Recall':>8} {'N':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for dim, m in results["dim_metrics"].items():
        print(f"  {dim:<30} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>8.3f} {m['n']:>6}")
    print("="*65)
    print()
    print("  Paper claims (vs 7B–27B autoregressive guardrails):")
    print("    • Competitive F1 on 9 established safety benchmarks")
    print("    • 16× higher throughput")
    print("    • 17× lower latency")
    print("    • 23–90× smaller model (0.3B vs 7B–27B)")


if __name__ == "__main__":
    main()
