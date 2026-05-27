"""
Evaluation script for ClaimDiff-RL.

Metrics computed:
  - Hallucination Rate: fraction of generated claims that are hallucinations
  - Omission Rate: fraction of reference claims missing from the candidate
  - ClaimDiff Score: composite quality score (higher = fewer errors)
  - Faithfulness-Coverage Balance: harmonic mean of (1 - hall_rate) and (1 - omit_rate)

Paper evaluation (Section 4):
  - 160-image human-labeled diagnostic benchmark
  - Public captioning benchmarks (CHAIR, POPE, etc.)
  - VQA benchmarks (MMStar, etc.)
  - Fine-grained capability dimensions vs. Gemini-3-Pro-Preview

This toy evaluates on the 5 toy samples using the ClaimDiff Judge.
"""
import argparse
import torch
import os
import sys

from data import ToyCapDataset, collate_fn
from judge import ClaimDiffJudge
from model import ToyCaptionPolicy, text_to_image_feat, decode
from reward import RewardConfig, compute_reward
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/claimdiff_rl.pt",
                   help="Path to trained model checkpoint")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.8)
    return p.parse_args()


def evaluate(model, dataset, judge, reward_cfg, device, temperature=0.8):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()

    total_hall = 0
    total_omit = 0
    total_samples = 0
    total_reward = 0.0
    results = []

    with torch.no_grad():
        for batch in loader:
            image_desc = batch["image_descs"][0]
            reference = batch["references"][0]

            img_feat = text_to_image_feat(image_desc, d_model=128, device=device).to(device)

            generated = model.generate(img_feat, max_new_tokens=24, temperature=temperature)
            candidate = decode(generated[0].tolist()) or "unknown"

            diffs = judge.judge(candidate, reference, image_desc)
            summary = judge.summarize(diffs)
            r = compute_reward(diffs, candidate, reference, reward_cfg)

            total_hall += summary["n_hallucinations"]
            total_omit += summary["n_omissions"]
            total_reward += r["R_total"]
            total_samples += 1

            results.append({
                "reference": reference,
                "candidate": candidate,
                "n_hall": summary["n_hallucinations"],
                "n_omit": summary["n_omissions"],
                "reward": r["R_total"],
            })

    hall_rate = total_hall / max(total_samples, 1)
    omit_rate = total_omit / max(total_samples, 1)
    avg_reward = total_reward / max(total_samples, 1)

    faithfulness = max(1.0 - hall_rate, 0.0)
    coverage = max(1.0 - omit_rate, 0.0)
    fc_balance = (2 * faithfulness * coverage) / max(faithfulness + coverage, 1e-8)

    return {
        "hallucination_rate": hall_rate,
        "omission_rate": omit_rate,
        "avg_reward": avg_reward,
        "faithfulness": faithfulness,
        "coverage": coverage,
        "faithfulness_coverage_balance": fc_balance,
        "results": results,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ToyCaptionPolicy().to(device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print(f"No checkpoint found at {args.ckpt}. Evaluating untrained model.")

    dataset = ToyCapDataset()
    judge = ClaimDiffJudge()
    reward_cfg = RewardConfig(alpha=args.alpha, beta=args.beta)

    metrics = evaluate(model, dataset, judge, reward_cfg, device, args.temperature)

    print("\n" + "=" * 60)
    print("ClaimDiff-RL Evaluation Results")
    print("=" * 60)
    print(f"  Hallucination Rate:            {metrics['hallucination_rate']:.3f}")
    print(f"  Omission Rate:                 {metrics['omission_rate']:.3f}")
    print(f"  Faithfulness (1 - hall_rate):  {metrics['faithfulness']:.3f}")
    print(f"  Coverage     (1 - omit_rate):  {metrics['coverage']:.3f}")
    print(f"  F-C Balance (harmonic mean):   {metrics['faithfulness_coverage_balance']:.3f}")
    print(f"  Average Reward:                {metrics['avg_reward']:.4f}")
    print()
    print("Per-sample results:")
    for i, r in enumerate(metrics["results"]):
        print(f"  [{i}] ref: {r['reference'][:50]}...")
        print(f"       gen: {r['candidate'][:50]}")
        print(f"       hall={r['n_hall']} omit={r['n_omit']} reward={r['reward']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
