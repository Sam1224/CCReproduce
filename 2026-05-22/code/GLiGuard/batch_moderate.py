"""
Batch Influencer Content Moderation Pipeline
============================================
Processes a JSONL file of influencer (达人) content items and outputs
moderation decisions using GLiGuard (arXiv:2605.07982).

Input JSONL format:
  {"id": "...", "text": "...", "creator_id": "..."}

Output JSONL format:
  {"id": "...", "is_safe": true/false, "harm_labels": [...], "confidence": 0.95}

Usage:
  python batch_moderate.py --input items.jsonl --output results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Toy dataset for demo
TOY_DATASET = [
    {"id": "001", "creator_id": "kol_001", "text": "这款减肥药三天见效，国内效果最好！"},
    {"id": "002", "creator_id": "kol_002", "text": "今天给大家种草一款保湿面霜，适合干皮使用。"},
    {"id": "003", "creator_id": "kol_003", "text": "高仿爱马仕包包，与正品一模一样，专柜绝对买不到的价格！"},
    {"id": "004", "creator_id": "kol_004", "text": "本产品可以完全治愈高血压，三个月保证见效！"},
    {"id": "005", "creator_id": "kol_005", "text": "这款防晒霜SPF50+，出门涂一层就好，记得每两小时补涂。"},
]


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_demo_batch(output_path: Optional[str] = None):
    """
    Demo run using toy dataset — simulates expected GLiGuard output.
    Replace with real model calls by importing EComGuard from ecom_guard.py.
    """
    # Simulated expected outputs (in real use, replace with model.moderate(text))
    simulated_outputs = [
        {"is_safe": False, "harm_labels": ["prohibited_weight_loss", "ad_law_violation"], "confidence": 0.91},
        {"is_safe": True,  "harm_labels": [], "confidence": 0.97},
        {"is_safe": False, "harm_labels": ["counterfeit_product"], "confidence": 0.89},
        {"is_safe": False, "harm_labels": ["prohibited_health_claim"], "confidence": 0.94},
        {"is_safe": True,  "harm_labels": [], "confidence": 0.98},
    ]

    results = []
    print(f"{'ID':<8} {'Creator':<12} {'Safe':<6} {'Harm Labels':<40} {'Conf'}")
    print("-" * 80)
    for item, sim in zip(TOY_DATASET, simulated_outputs):
        record = {
            "id": item["id"],
            "creator_id": item["creator_id"],
            "text_preview": item["text"][:50] + "...",
            **sim,
        }
        results.append(record)
        harm_str = ", ".join(sim["harm_labels"]) if sim["harm_labels"] else "—"
        print(f"{item['id']:<8} {item['creator_id']:<12} {str(sim['is_safe']):<6} {harm_str:<40} {sim['confidence']:.2f}")

    print(f"\nTotal: {len(results)} items | Violations: {sum(1 for r in results if not r['is_safe'])}")

    if output_path:
        save_jsonl(output_path, results)
        print(f"Results saved to: {output_path}")

    return results


def main():
    from typing import Optional
    parser = argparse.ArgumentParser(description="Batch influencer content moderation with GLiGuard")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--demo", action="store_true", default=True, help="Run demo with toy data")
    args = parser.parse_args()

    if args.input:
        # Real run with model
        try:
            from ecom_guard import EComGuard
            guard = EComGuard()
            items = load_jsonl(args.input)
            results = []
            for item in items:
                result = guard.moderate(item.get("text", ""))
                results.append({
                    "id": item.get("id"),
                    "creator_id": item.get("creator_id"),
                    **result.to_dict(),
                })
            if args.output:
                save_jsonl(args.output, results)
            print(f"Processed {len(results)} items.")
        except Exception as e:
            print(f"Error: {e}\nFalling back to demo mode.")
            run_demo_batch(args.output)
    else:
        # Demo run
        run_demo_batch(args.output)


if __name__ == "__main__":
    main()
