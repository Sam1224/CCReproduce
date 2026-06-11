"""
QueryAgent-R1 RL training script.

Training paradigm (paper §3.4):
  - GRPO-style group reward: generate G candidate queries per user,
    compute consistency rewards, normalize within the group, and use
    the advantage-weighted policy gradient loss.

  Formula (GRPO objective, adapted from paper):
    L_GRPO = -E[ Σ_t log π_θ(a_t | s_t) * A_i ]
    where A_i = (r_i - mean(r)) / std(r)  — normalized group advantage

  We also optionally add a SFT warmup loss on positive query demonstrations.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List

from data import build_datasets, QuerySample
from model import QueryAgentR1, consistency_reward


def sample_history_tensors(
    sample: QuerySample,
    user_dataset,
    catalog,
    device: torch.device,
    click_seq_len: int = 10,
    purch_seq_len: int = 3,
) -> tuple:
    """Convert a QuerySample to model-ready tensors."""
    user = user_dataset.get_user(sample.user_id)
    if user is None:
        # Fallback: random embeddings
        click_emb = torch.randn(click_seq_len, catalog.embedding_dim)
        purch_emb = torch.randn(purch_seq_len, catalog.embedding_dim)
        return click_emb.to(device), purch_emb.to(device)

    # Retrieve actual product embeddings
    click_ids = user.clicked_product_ids[:click_seq_len]
    purch_ids = user.purchased_product_ids[:purch_seq_len]

    def get_embs(ids, length):
        products = catalog.get_by_ids(ids)
        embs = [torch.tensor(p.embedding, dtype=torch.float32) for p in products]
        # Pad to fixed length
        while len(embs) < length:
            embs.append(torch.zeros(catalog.embedding_dim, dtype=torch.float32))
        return torch.stack(embs[:length])

    return get_embs(click_ids, click_seq_len).to(device), \
           get_embs(purch_ids, purch_seq_len).to(device)


def collate_batch(
    samples: List[QuerySample],
    user_dataset,
    catalog,
    device: torch.device,
) -> tuple:
    """Collate a list of samples into batched tensors."""
    click_embs, purch_embs = [], []
    for s in samples:
        c, p = sample_history_tensors(s, user_dataset, catalog, device)
        click_embs.append(c)
        purch_embs.append(p)
    return torch.stack(click_embs), torch.stack(purch_embs)


def grpo_loss(
    model: QueryAgentR1,
    click_embs: torch.Tensor,
    purch_embs: torch.Tensor,
    catalog,
    n_samples: int = 4,
    temperature: float = 1.2,
) -> torch.Tensor:
    """
    GRPO-style policy gradient loss.

    For each user in the batch:
      1. Sample G=n_samples queries
      2. Compute consistency reward for each
      3. Normalize rewards within group → advantages
      4. Compute policy gradient loss: -log_prob * advantage
    """
    B = click_embs.shape[0]
    device = click_embs.device

    all_rewards = []
    all_log_probs = []

    for _ in range(n_samples):
        # Forward pass to get rewards and query log probs
        context, hist_emb, purch_emb = model.encode_history(click_embs, purch_embs)

        feedback = None
        query_ids, query_emb, retrieved, feedback = model.generate_and_retrieve(
            context, catalog, feedback, temperature
        )

        # Compute log probs of generated tokens
        # Teacher-forced forward pass using generated ids as input
        tgt_input = query_ids[:, :-1]
        tgt_target = query_ids[:, 1:]
        logits = model.policy_lm(tgt_input, context, feedback)  # (B, T-1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, tgt_target.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)
        # Mask EOS tokens
        eos_mask = (tgt_target != model.policy_lm.EOS_ID).float()
        seq_log_prob = (token_log_probs * eos_mask).sum(1) / (eos_mask.sum(1) + 1e-8)  # (B,)

        # Compute reward
        ret_emb_list = []
        for batch_results in retrieved:
            embs = np.stack([p.embedding for p in batch_results])
            ret_emb_list.append(embs.mean(0))
        ret_emb = torch.tensor(
            np.stack(ret_emb_list), dtype=torch.float32, device=device
        )
        reward = consistency_reward(query_emb, hist_emb, ret_emb, purch_emb)

        all_rewards.append(reward)
        all_log_probs.append(seq_log_prob)

    # Stack: (n_samples, B)
    rewards = torch.stack(all_rewards, dim=0)
    log_probs_stack = torch.stack(all_log_probs, dim=0)

    # Normalize advantages within group for each user
    mean_r = rewards.mean(dim=0, keepdim=True)
    std_r = rewards.std(dim=0, keepdim=True) + 1e-8
    advantages = (rewards - mean_r) / std_r  # (n_samples, B)

    # Policy gradient loss: -E[log_prob * advantage]
    loss = -(log_probs_stack * advantages.detach()).mean()
    return loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    catalog, users, train_set, eval_set = build_datasets(
        n_products=args.n_products,
        n_users=args.n_users,
        n_train=args.n_train,
        n_eval=args.n_eval,
        embedding_dim=args.emb_dim,
        seed=args.seed,
    )

    # Model
    model = QueryAgentR1(
        catalog_emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_refinement_steps=args.n_refine,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs("checkpoints", exist_ok=True)
    best_reward = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_reward = 0.0
        n_batches = 0

        # Shuffle train indices
        indices = torch.randperm(len(train_set)).tolist()
        for start in tqdm(range(0, len(indices), args.batch_size),
                          desc=f"Epoch {epoch}", leave=False):
            batch_ids = indices[start:start + args.batch_size]
            batch_samples = [train_set[i] for i in batch_ids]

            click_embs, purch_embs = collate_batch(batch_samples, users, catalog, device)

            loss = grpo_loss(
                model, click_embs, purch_embs, catalog,
                n_samples=args.n_samples, temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track mean reward for logging
            with torch.no_grad():
                out = model(click_embs, purch_embs, catalog)
            total_loss += loss.item()
            total_reward += out["reward"].mean().item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_reward = total_reward / max(n_batches, 1)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | reward={avg_reward:.4f}")

        # Eval
        eval_reward = evaluate(model, eval_set, users, catalog, device, args)
        print(f"          | eval_reward={eval_reward:.4f}")

        if eval_reward > best_reward:
            best_reward = eval_reward
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print(f"          | ✓ saved best (reward={best_reward:.4f})")

    print(f"\nTraining complete. Best eval reward: {best_reward:.4f}")


@torch.no_grad()
def evaluate(model, eval_set, users, catalog, device, args) -> float:
    model.eval()
    total_reward = 0.0
    n_batches = 0
    indices = list(range(len(eval_set)))
    for start in range(0, len(indices), args.batch_size):
        batch_ids = indices[start:start + args.batch_size]
        batch_samples = [eval_set[i] for i in batch_ids]
        click_embs, purch_embs = collate_batch(batch_samples, users, catalog, device)
        out = model(click_embs, purch_embs, catalog)
        total_reward += out["reward"].mean().item()
        n_batches += 1
    return total_reward / max(n_batches, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QueryAgent-R1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_products", type=int, default=2000)
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_refine", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
