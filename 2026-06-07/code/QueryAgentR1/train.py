"""
Training script for QueryAgent-R1 reproduction.
Two-stage training:
  Stage 1: Supervised fine-tuning (SFT) of query generator with teacher forcing
  Stage 2: GRPO-based RL fine-tuning with consistency reward

Usage:
    python train.py --epochs 10 --rl_epochs 5 --batch_size 16
"""

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import QueryAgentR1
from reward import ConsistencyReward, GRPOLoss, CVRProxyReward
from data import (
    get_dataloaders,
    get_product_corpus,
    VOCAB_SIZE,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SimpleTokenizer,
)
from retrieval import BM25Retriever
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="QueryAgent-R1 Training")
    parser.add_argument("--item_embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_interest_nodes", type=int, default=4)
    parser.add_argument("--max_query_len", type=int, default=16)
    parser.add_argument("--num_gen_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10, help="SFT epochs")
    parser.add_argument("--rl_epochs", type=int, default=5, help="RL fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rl_lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.5, help="CTR reward weight")
    parser.add_argument("--beta", type=float, default=0.5, help="CVR reward weight")
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sft_train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for batch in loader:
        hist_embeds = batch["hist_embeds"].to(device)
        hist_mask = batch["hist_mask"].to(device)
        query_ids = batch["query_ids"].to(device)  # (B, T)

        # Teacher forcing: input is query[:-1], target is query[1:]
        input_ids = query_ids[:, :-1]
        target_ids = query_ids[:, 1:]

        outputs = model(
            item_embeds=hist_embeds,
            query_input_ids=input_ids,
            item_mask=hist_mask,
        )
        logits = outputs["logits"]  # (B, T-1, V)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            target_ids.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def rl_train_epoch(
    model,
    ref_model,
    loader,
    optimizer,
    consistency_reward,
    grpo_loss_fn,
    retriever,
    tokenizer,
    device,
    top_k: int = 5,
):
    """
    GRPO-based RL fine-tuning with consistency reward.

    For each batch:
    1. Generate query using current policy
    2. Retrieve products using BM25
    3. Compute consistency reward
    4. Compute GRPO loss and update policy
    """
    model.train()
    ref_model.eval()

    total_loss = 0.0
    total_reward = 0.0
    n_batches = 0

    for batch in loader:
        hist_embeds = batch["hist_embeds"].to(device)
        hist_mask = batch["hist_mask"].to(device)
        pos_prod_embed = batch["pos_prod_embed"].to(device)  # (B, D)

        B = hist_embeds.shape[0]

        # Encode user
        with torch.no_grad():
            user_embed, _ = model.encode_user(hist_embeds, hist_mask)

        # Generate queries using current policy (no teacher forcing)
        model.eval()
        with torch.no_grad():
            gen_ids = model.generate_query(
                user_embed=user_embed,
                bos_token_id=BOS_ID,
                eos_token_id=EOS_ID,
                max_new_tokens=model.generator.max_query_len - 2,
            )

        model.train()

        # Compute log probs of generated queries under current and reference policy
        gen_input = gen_ids[:, :-1]
        gen_target = gen_ids[:, 1:]

        outputs = model(hist_embeds, gen_input, hist_mask)
        logits = outputs["logits"]  # (B, T, V)
        log_probs_all = F.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(
            -1, gen_target.clamp(0, VOCAB_SIZE - 1).unsqueeze(-1)
        ).squeeze(-1)
        # Mean log prob over non-pad tokens
        valid_mask = (gen_target != PAD_ID).float()
        log_probs_mean = (log_probs * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-8)

        with torch.no_grad():
            ref_out = ref_model(hist_embeds, gen_input, hist_mask)
            ref_logits = ref_out["logits"]
            ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs_all.gather(
                -1, gen_target.clamp(0, VOCAB_SIZE - 1).unsqueeze(-1)
            ).squeeze(-1)
            ref_log_probs_mean = (ref_log_probs * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-8)

        # Compute consistency reward
        # query_embed = user_embed as proxy (in practice, encode generated query separately)
        query_embed = user_embed  # simplification: use user embed as query embed proxy

        # Toy retrieved product embeddings (in practice, use dense retrieval)
        # Here we use a random subset of product embeds as "retrieved"
        from data import ECommerceQueryDataset
        toy_ds = ECommerceQueryDataset(num_users=1, item_embed_dim=hist_embeds.shape[-1])
        all_prod_embeds = toy_ds.product_embeds.to(device)  # (N, D)
        retrieved_prod_embeds = all_prod_embeds[:top_k].unsqueeze(0).expand(B, -1, -1)

        with torch.no_grad():
            reward_dict = consistency_reward(
                query_embed=query_embed,
                user_embed=user_embed,
                retrieved_product_embeds=retrieved_prod_embeds,
                positive_product_embeds=pos_prod_embed.unsqueeze(1),
            )
        rewards = reward_dict["total_reward"].detach()

        # GRPO loss
        loss_dict = grpo_loss_fn(
            log_probs=log_probs_mean,
            old_log_probs=log_probs_mean.detach(),  # same step (simplified)
            rewards=rewards,
            ref_log_probs=ref_log_probs_mean,
        )
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reward += rewards.mean().item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_reward / max(n_batches, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=== QueryAgent-R1 Training (Toy Reproduction) ===")
    print(f"Device: {device} | Hidden: {args.hidden_dim} | Nodes: {args.num_interest_nodes}")

    # Data
    train_loader, val_loader = get_dataloaders(
        num_users=200,
        history_len=20,
        item_embed_dim=args.item_embed_dim,
        batch_size=args.batch_size,
    )
    tokenizer = SimpleTokenizer()

    # Build retriever
    retriever = BM25Retriever()
    retriever.build(get_product_corpus())

    # Model
    model = QueryAgentR1(
        item_embed_dim=args.item_embed_dim,
        vocab_size=VOCAB_SIZE,
        num_interest_nodes=args.num_interest_nodes,
        hidden_dim=args.hidden_dim,
        max_query_len=args.max_query_len,
        num_gen_layers=args.num_gen_layers,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ============================
    # Stage 1: SFT
    # ============================
    print("\n--- Stage 1: SFT ---")
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_loss = sft_train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        metrics = evaluate(model, val_loader, retriever, tokenizer, device)
        print(
            f"  Epoch {epoch+1:02d}/{args.epochs} | "
            f"SFT Loss: {train_loss:.4f} | "
            f"Val MRR@5: {metrics['mrr@5']:.4f} | "
            f"Val CTR: {metrics['ctr_proxy']:.4f}"
        )

    # Save SFT checkpoint
    torch.save(model.state_dict(), "checkpoints_sft.pt")
    print("SFT checkpoint saved.")

    # ============================
    # Stage 2: RL Fine-tuning
    # ============================
    print("\n--- Stage 2: GRPO RL Fine-tuning ---")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    query_encoder_proxy = model.memory  # use memory as a proxy for query encoder
    consistency_reward = ConsistencyReward(
        query_encoder=query_encoder_proxy,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        beta=args.beta,
    ).to(device)
    grpo_loss_fn = GRPOLoss(clip_epsilon=args.clip_epsilon, kl_coef=args.kl_coef)

    rl_optimizer = AdamW(model.parameters(), lr=args.rl_lr)

    for epoch in range(args.rl_epochs):
        rl_loss, mean_reward = rl_train_epoch(
            model, ref_model, train_loader, rl_optimizer,
            consistency_reward, grpo_loss_fn, retriever, tokenizer, device,
        )
        metrics = evaluate(model, val_loader, retriever, tokenizer, device)
        print(
            f"  RL Epoch {epoch+1:02d}/{args.rl_epochs} | "
            f"RL Loss: {rl_loss:.4f} | "
            f"Mean Reward: {mean_reward:.4f} | "
            f"Val MRR@5: {metrics['mrr@5']:.4f}"
        )

    torch.save(model.state_dict(), "checkpoints_rl.pt")
    print("RL checkpoint saved.")
    print("\nTraining complete!")


if __name__ == "__main__":
    import os
    os.makedirs(".", exist_ok=True)
    main()
