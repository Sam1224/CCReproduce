"""
GRPO RL Training for QueryAgent-R1
Paper: arxiv 2606.05671

Training protocol:
  Phase 1 (SFT cold-start): supervised fine-tuning on target queries
  Phase 2 (GRPO RL):        reinforcement learning with consistency reward

GRPO specifics for QueryAgent-R1:
  - Group size G = 8 (sample 8 queries per user-intent prompt)
  - Reward = α * consistency_reward + β * relevance_reward
  - Reference policy = SFT model (KL penalty prevents reward hacking)
  - Update: PPO-clip style with group-normalized advantages
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Tuple

from model import QueryAgent, GRPOTrainer
from retrieval import ProductRetriever, UserHistoryEncoder
from data import EcommerceQueryDataset, get_dataloader


def sft_warmup(
    agent: QueryAgent,
    train_loader,
    args: argparse.Namespace,
) -> None:
    """
    Phase 1: SFT on expert query annotations.
    Gives RL training a stable starting policy.
    """
    print("Phase 1: SFT cold-start training")
    optimizer = AdamW(agent.parameters(), lr=args.sft_lr, weight_decay=1e-4)

    for epoch in range(args.sft_epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            # SFT loss: cross-entropy on target query tokens
            # (pseudocode — actual implementation needs LLM forward pass)
            # loss = llm_crossentropy(target_query_tokens)
            loss = torch.tensor(0.5 - 0.1 * epoch + 0.01 * step % 10)  # toy stub
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  [SFT] Epoch {epoch+1} Step {step} Loss={loss.item():.4f}")

        print(f"SFT Epoch {epoch+1} avg loss: {total_loss/max(len(train_loader),1):.4f}")


def rl_training_step(
    agent: QueryAgent,
    grpo: GRPOTrainer,
    retriever: ProductRetriever,
    batch: Dict,
    device: str,
) -> Tuple[torch.Tensor, Dict]:
    """
    Single GRPO training step.
    
    1. Sample G queries for each prompt (user intent + memory context)
    2. Retrieve products for each query
    3. Compute rewards
    4. Compute GRPO loss with group-normalized advantages
    
    Returns:
        loss: GRPO loss tensor
        metrics: dict with reward stats
    """
    B = batch["history_categories"].shape[0]
    G = grpo.group_size

    # Encode user memory
    interest_embs, top_k_interests = agent.memory_module(
        batch["history_categories"].to(device),
        batch["history_behaviors"].to(device),
        batch["history_weights"].to(device),
        batch["history_mask"].to(device),
    )  # [B, embed_dim], List[List[Dict]]

    all_rewards = []
    all_log_probs = []
    all_ref_log_probs = []

    for b in range(B):
        user_interest = interest_embs[b]  # [embed_dim]
        search_intent = batch["search_intent"][b]
        interests = top_k_interests[b]

        group_rewards = []
        group_log_probs = []

        for g in range(G):
            # Generate query (toy: random variation; real: LLM sample)
            base_query = batch["target_query"][b]
            toy_variants = [
                f"buy {base_query}", f"best {base_query}", f"cheap {base_query}",
                f"{base_query} discount", f"{base_query} top rated",
                f"new {base_query}", f"{base_query} sale", f"find {base_query}",
            ]
            generated_query = toy_variants[g % len(toy_variants)]

            # Retrieve products
            products = retriever.retrieve(generated_query, top_k=10)

            # Compute rewards
            prod_embs = torch.stack([
                retriever.encode_query(p["title"]) for p in products
            ])  # [K, embed_dim]

            cons_reward = agent.compute_consistency_reward(prod_embs, user_interest)

            # Query embedding (toy: from query hash)
            query_emb = retriever.encode_query(generated_query)
            intent_emb = retriever.encode_query(search_intent)
            rel_reward = agent.compute_relevance_reward(query_emb, intent_emb)

            total_reward = agent.compute_total_reward(
                cons_reward, rel_reward, alpha=0.6, beta=0.4
            )
            group_rewards.append(total_reward)

            # Log prob (toy: uniform; real: from LLM logits)
            log_prob = torch.tensor(-2.0 + 0.1 * g)  # toy stub
            group_log_probs.append(log_prob)

        group_rewards = torch.stack(group_rewards)
        group_log_probs = torch.stack(group_log_probs)
        group_ref_log_probs = group_log_probs - 0.1  # toy ref policy

        all_rewards.append(group_rewards)
        all_log_probs.append(group_log_probs)
        all_ref_log_probs.append(group_ref_log_probs)

    # Flatten over batch and group for GRPO loss
    all_rewards = torch.cat(all_rewards)
    all_log_probs = torch.cat(all_log_probs)
    all_ref_log_probs = torch.cat(all_ref_log_probs)

    loss = grpo.grpo_loss(all_log_probs, all_ref_log_probs, all_rewards)

    metrics = {
        "reward_mean": all_rewards.mean().item(),
        "reward_std": all_rewards.std().item(),
        "loss": loss.item(),
    }
    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train QueryAgent-R1 with GRPO")
    parser.add_argument("--data", default=None, help="Data directory (None = toy)")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--rl_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sft_lr", type=float, default=1e-4)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--toy", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    print(f"QueryAgent-R1 Training | device={device} | toy={args.toy}")

    # Initialize components
    agent = QueryAgent(embed_dim=256, num_categories=20, max_refine_steps=3)
    retriever = ProductRetriever(embed_dim=256, catalog_size=1000 if args.toy else 10000)
    grpo = GRPOTrainer(agent, retriever, group_size=args.group_size)

    train_loader = get_dataloader(
        args.data, split="train",
        batch_size=args.batch_size,
        toy_size=200 if args.toy else 100000,
    )
    val_loader = get_dataloader(
        args.data, split="val",
        batch_size=args.batch_size,
        toy_size=50 if args.toy else 10000,
    )

    optimizer = AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)

    # Phase 1: SFT cold-start
    sft_warmup(agent, train_loader, args)

    # Phase 2: GRPO RL training
    print("\nPhase 2: GRPO RL training")
    best_reward = -float("inf")

    for epoch in range(args.rl_epochs):
        agent.train()
        epoch_rewards = []

        for step, batch in enumerate(train_loader):
            loss, metrics = rl_training_step(agent, grpo, retriever, batch, device)
            epoch_rewards.append(metrics["reward_mean"])

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if step % 20 == 0:
                print(f"  [RL] Epoch {epoch+1}/{args.rl_epochs} "
                      f"Step {step} | reward={metrics['reward_mean']:.4f} "
                      f"std={metrics['reward_std']:.4f} loss={metrics['loss']:.4f}")

        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        print(f"Epoch {epoch+1} avg reward: {avg_reward:.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(agent.state_dict(),
                       f"{args.output_dir}/queryagent_r1_best.pt")
            print(f"  → Saved best model (reward={best_reward:.4f})")

    print(f"\nTraining complete. Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
