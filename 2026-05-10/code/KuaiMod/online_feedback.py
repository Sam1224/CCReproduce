"""
KuaiMod Online Feedback Refinement Loop

Simulates the online policy update mechanism (Section 3.3 of the paper).

In production (Kuaishou):
  - User reports → high-confidence violations → added to training pool
  - Reviewer corrections → hard cases for curriculum learning
  - Policy updates → new rule injection via CoT template updates

This script simulates this with a static validation set split
into "user_report", "reviewer", and "policy_update" batches.

The paper measures:
  - Policy update latency (how quickly new rules are learned)
  - User reporting rate reduction (production metric: -20%)
"""

import argparse
import random
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data import SVPDataset, SVPCollator, LABEL2ID, ID2LABEL
from model import KuaiMod
from train import simple_tokenize, simple_tokenize_cot, compute_metrics


class FeedbackBuffer:
    """
    Simulates the online feedback buffer.
    In production: receives live user reports + reviewer labels.
    Here: selects samples by feedback_source from static dataset.
    """

    def __init__(self, dataset: SVPDataset, max_size: int = 100):
        self.buffer = []
        self.max_size = max_size
        self._populate(dataset)

    def _populate(self, dataset):
        for sample in dataset.samples:
            self.buffer.append(sample)
        random.shuffle(self.buffer)
        self.buffer = self.buffer[: self.max_size]

    def get_by_source(self, source: str):
        return [s for s in self.buffer if s.feedback_source == source]

    def add(self, sample):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(sample)


def online_update_step(model, samples, optimizer, device, vocab_size=1000):
    """
    Single gradient step on a batch of feedback samples.
    Paper uses a small learning rate with warm restart for online updates.
    """
    if not samples:
        return None

    collator = SVPCollator()
    batch = collator(samples[:min(len(samples), 8)])
    frames = batch["frames"].to(device)
    texts = batch["texts"]
    labels = batch["labels"].to(device)
    cot_texts = batch["cot_rationales"]

    text_ids = simple_tokenize(texts, vocab_size).to(device)
    cot_ids = simple_tokenize_cot(cot_texts, vocab_size).to(device)

    model.train()
    optimizer.zero_grad()
    out = model(frames, text_ids, cot_ids=cot_ids, labels=labels, stage="cot")
    loss = out["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def simulate_user_reporting_rate(model, dataset, device, vocab_size=1000):
    """
    Simulate user reporting rate reduction.
    Proxy: fraction of ground-truth violations that the model MISSES.
    Paper reports -20% reduction in user reporting rate after deployment.
    """
    model.eval()
    collator = SVPCollator()
    all_samples = dataset.samples
    batch = collator(all_samples[:64])
    frames = batch["frames"].to(device)
    texts = batch["texts"]
    labels_list = [s.label for s in all_samples[:64]]

    text_ids = simple_tokenize(texts, vocab_size).to(device)
    pred = model.predict(frames, text_ids)
    verdicts = pred["verdicts"].cpu().tolist()

    # Miss rate = FN / (TP + FN) for 'violating' class
    fn = sum(1 for p, l in zip(verdicts, labels_list) if p != 1 and l == 1)
    positives = sum(1 for l in labels_list if l == 1)
    miss_rate = fn / max(positives, 1)
    return miss_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of online update rounds")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--lr_online", type=float, default=1e-5,
                        help="Low LR for online refinement (paper recommendation)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SVPDataset(num_samples=300, seed=99)
    model = KuaiMod(vocab_size=args.vocab_size).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded: {args.checkpoint}")

    # Small LR for online updates (avoid catastrophic forgetting)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_online, weight_decay=0)

    feedback_buffer = FeedbackBuffer(dataset, max_size=150)

    print(f"\n=== Online Feedback Refinement Simulation ===")
    print(f"Rounds: {args.num_rounds} | Buffer size: {len(feedback_buffer.buffer)}")

    baseline_miss = simulate_user_reporting_rate(model, dataset, device, args.vocab_size)
    print(f"Baseline miss rate (proxy for user reporting): {baseline_miss:.3f}")

    for round_idx in range(1, args.num_rounds + 1):
        # Prioritize reviewer feedback (harder cases)
        reviewer_samples = feedback_buffer.get_by_source("reviewer")
        user_samples = feedback_buffer.get_by_source("user_report")
        policy_samples = feedback_buffer.get_by_source("policy_update")

        loss_r = online_update_step(model, reviewer_samples, optimizer, device,
                                    args.vocab_size)
        loss_u = online_update_step(model, user_samples, optimizer, device,
                                    args.vocab_size)
        loss_p = online_update_step(model, policy_samples, optimizer, device,
                                    args.vocab_size)

        miss_rate = simulate_user_reporting_rate(model, dataset, device, args.vocab_size)

        losses_str = (
            f"reviewer={loss_r:.4f}" if loss_r else "reviewer=N/A",
            f"user={loss_u:.4f}" if loss_u else "user=N/A",
            f"policy={loss_p:.4f}" if loss_p else "policy=N/A",
        )
        print(
            f"Round {round_idx}/{args.num_rounds} | "
            f"losses: {', '.join(losses_str)} | "
            f"miss rate: {miss_rate:.3f} "
            f"(Δ={miss_rate - baseline_miss:+.3f})"
        )

    final_miss = simulate_user_reporting_rate(model, dataset, device, args.vocab_size)
    improvement = (baseline_miss - final_miss) / max(baseline_miss, 1e-6) * 100
    print(f"\nFinal miss rate: {final_miss:.3f}")
    print(f"Simulated reporting rate reduction: {improvement:.1f}%")
    print(f"(Paper reports -20% reduction in production; random weights expected ~0%)")


if __name__ == "__main__":
    main()
