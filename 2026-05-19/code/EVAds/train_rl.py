"""Training script for E-VAds-R1 with RL fine-tuning.

Implements the paper's training recipe:
  1. Start from base MLLM
  2. Fine-tune on E-VAds commercial intent Q&A with GRPO
  3. Few-shot regime: only few hundred training samples
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data.synthetic_evads import generate_evads_dataset
from model.evads_r1 import GRPOTrainer, ToyMLLM


class EVAdsDataset(Dataset):
    """Simplified dataset for E-VAds commercial intent Q&A."""

    # Canonical answer vocabulary (simplified)
    ANSWER_VOCAB = [
        "skincare", "fashion", "electronics", "food & beverage",
        "home appliances", "sports equipment", "books", "toys",
        "direct product promotion", "affiliate marketing",
        "brand awareness", "limited-time offer push",
        "lifestyle endorsement", "tutorial with product placement",
        "unboxing/review", "emotional appeal", "scarcity/urgency",
        "social proof (reviews)", "celebrity/KOL endorsement",
        "before-after demonstration",
    ]
    WORD_VOCAB_SIZE = 5000

    def __init__(self, samples, max_len: int = 64, target_type: str = "commercial_intent"):
        self.samples = samples
        self.max_len = max_len
        self.target_type = target_type
        self.ans2id = {a: i for i, a in enumerate(self.ANSWER_VOCAB)}

    def _tokenize(self, text: str) -> list[int]:
        tokens = [hash(w) % (self.WORD_VOCAB_SIZE - 2) + 2 for w in text.lower().split()]
        return tokens[:self.max_len]

    def _pad(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        pad_len = self.max_len - len(tokens)
        mask = [1] * len(tokens) + [0] * pad_len
        tokens = tokens + [0] * pad_len
        return tokens, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Build input: description + question about commercial intent
        qa = next(
            (q for q in sample.questions if q["question_type"] == self.target_type),
            sample.questions[0],
        )
        text = f"{sample.video_description} Q: {qa['question']}"
        tokens = self._tokenize(text)
        ids, mask = self._pad(tokens)

        # Answer class
        answer = qa["answer"]
        label = self.ans2id.get(answer, 0)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAds-R1] Training on {device}")

    # Generate toy dataset (simulates few-hundred training samples from E-VAds)
    all_samples = generate_evads_dataset(num_videos=args.num_train_samples + 50, seed=args.seed)
    train_samples = all_samples[:args.num_train_samples]
    val_samples = all_samples[args.num_train_samples:]

    train_ds = EVAdsDataset(train_samples)
    val_ds = EVAdsDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize base model and reference model
    model = ToyMLLM(
        vocab_size=EVAdsDataset.WORD_VOCAB_SIZE,
        num_answer_classes=len(EVAdsDataset.ANSWER_VOCAB),
    )
    ref_model = copy.deepcopy(model)

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        lr=args.lr,
        beta_kl=args.beta_kl,
        device=device,
    )

    best_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_reward = 0.0

        for batch in train_loader:
            metrics = trainer.grpo_step(
                batch["input_ids"],
                batch["attention_mask"],
                batch["label"],
            )
            total_loss += metrics["loss"]
            total_reward += metrics["mean_reward"]

        avg_loss = total_loss / len(train_loader)
        avg_reward = total_reward / len(train_loader)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                preds = logits.argmax(dim=-1).cpu()
                correct += (preds == batch["label"]).sum().item()
                total += len(batch["label"])
        val_acc = correct / max(total, 1)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        history.append({"epoch": epoch + 1, "loss": avg_loss, "reward": avg_reward, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[EVAds-R1] Training complete. Best val acc: {best_acc:.4f}")
    print(f"[EVAds-R1] Artifacts saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_samples", type=int, default=200,
                        help="Number of training samples (paper: few hundred)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_kl", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="runs/evads_r1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
