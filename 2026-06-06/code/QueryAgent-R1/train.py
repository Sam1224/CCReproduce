"""
QueryAgent-R1 Training Script

Phases:
    Phase 1 (SFT warmup): train query generator on human-curated (context, query) pairs
    Phase 2 (RL):         train with REINFORCE using consistency reward (CTR + CVR)

Usage:
    python train.py --phase sft --epochs 3
    python train.py --phase rl  --epochs 10
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.query_agent import QueryAgentR1
from retrieval.retriever import ProductRetriever
from rl.reward import ConsistencyReward
from rl.trainer import REINFORCETrainer
from data.dataset import EComQueryDataset


def sft_warmup(agent: QueryAgentR1, dataloader: DataLoader, epochs: int, device: str):
    """Phase 1: Supervised pre-training on (context, query) pairs."""
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    agent.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            hist = batch["interaction_history"].to(device)
            ctx = batch["current_context"].to(device)
            q_ids = batch["query_ids"].to(device)

            optimizer.zero_grad()
            out = agent(hist, ctx, tgt_ids=q_ids)
            logits = out["query_logits"]  # (B, L, V)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = q_ids[:, 1:].contiguous().view(-1)
            loss = criterion(shift_logits, shift_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"[SFT] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")


def rl_training(
    agent: QueryAgentR1,
    retriever: ProductRetriever,
    reward_fn: ConsistencyReward,
    dataloader: DataLoader,
    epochs: int,
    device: str,
):
    """Phase 2: RL training with chain-of-retrieval consistency reward."""
    trainer = REINFORCETrainer(agent, retriever, reward_fn, lr=2e-5, device=device)

    for epoch in range(epochs):
        metrics = {"total_loss": 0.0, "mean_reward": 0.0, "mean_ctr": 0.0, "mean_cvr": 0.0}
        n = 0
        for batch in dataloader:
            step_metrics = trainer.train_step(batch)
            for k in metrics:
                metrics[k] += step_metrics[k]
            n += 1

        print(
            f"[RL] Epoch {epoch+1}/{epochs} | "
            f"Loss: {metrics['total_loss']/n:.4f} | "
            f"Reward: {metrics['mean_reward']/n:.4f} | "
            f"CTR: {metrics['mean_ctr']/n:.4f} | "
            f"CVR: {metrics['mean_cvr']/n:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="sft", choices=["sft", "rl"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    dataset = EComQueryDataset(size=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    agent = QueryAgentR1(item_embed_dim=64, vocab_size=10_000, memory_len=20)
    retriever = ProductRetriever(vocab_size=10_000, embed_dim=64, num_products=5_000)
    reward_fn = ConsistencyReward(embed_dim=64, alpha=0.4)

    if args.phase == "sft":
        print("=== Phase 1: SFT Warmup ===")
        sft_warmup(agent, dataloader, args.epochs, device)
    else:
        print("=== Phase 2: RL Training ===")
        rl_training(agent, retriever, reward_fn, dataloader, args.epochs, device)

    print("Training complete.")


if __name__ == "__main__":
    main()
