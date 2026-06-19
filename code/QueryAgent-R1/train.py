"""
Training script for QueryAgent-R1.
Stage 1: SFT warm-up (teacher forcing)
Stage 2: GRPO RL fine-tuning (chain-of-retrieval + consistency reward)
"""

import copy
import torch
import argparse
from model import QueryAgentR1, QueryAgentConfig
from grpo_trainer import GRPOTrainer
from data import get_dataloader


def sft_train(model, dataloader, epochs: int = 3, lr: float = 1e-4, device: str = "cpu"):
    """Stage 1: Supervised fine-tuning with teacher forcing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    # Toy product catalog: N random product embeddings
    N = 10000
    catalog_emb = torch.randn(N, model.config.hidden_size, device=device)
    catalog_emb = torch.nn.functional.normalize(catalog_emb, dim=-1)

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            item_ids = batch["item_ids"].to(device)
            behavior_types = batch["behavior_types"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tgt_queries = batch["target_queries"].to(device)
            purchased_ids = batch["purchased_ids"].to(device)

            out = model(
                item_ids, behavior_types, attention_mask,
                tgt_queries, catalog_emb, purchased_ids
            )
            loss = out["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if step % 20 == 0:
                print(f"[SFT] Epoch {epoch+1} Step {step} loss={loss.item():.4f} "
                      f"consistency_reward={out['consistency_reward'].item():.4f}")

        print(f"[SFT] Epoch {epoch+1} avg_loss={total_loss / max(len(dataloader), 1):.4f}")


def rl_train(model, ref_model, dataloader, epochs: int = 2, device: str = "cpu"):
    """Stage 2: GRPO reinforcement learning fine-tuning."""
    trainer = GRPOTrainer(model, ref_model, lr=1e-5, group_size=4)
    model.train()
    N = 10000
    catalog_emb = torch.randn(N, model.config.hidden_size, device=device)
    catalog_emb = torch.nn.functional.normalize(catalog_emb, dim=-1)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            B = batch["item_ids"].size(0)
            G = trainer.group_size
            item_ids = batch["item_ids"].to(device)
            behavior_types = batch["behavior_types"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            purchased_ids = batch["purchased_ids"].to(device)
            # Toy CTR labels
            ctr_labels = torch.randint(0, 2, (B * G,), device=device).float()

            metrics = trainer.step(
                item_ids, behavior_types, attention_mask,
                catalog_emb, purchased_ids, ctr_labels
            )

            if step % 20 == 0:
                print(f"[RL] Epoch {epoch+1} Step {step} "
                      f"loss={metrics['loss']:.4f} "
                      f"reward={metrics['reward_mean']:.4f} "
                      f"consistency={metrics['consistency_reward']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--rl_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    config = QueryAgentConfig(
        vocab_size=32000, hidden_size=256, num_layers=3, num_heads=8, max_seq_len=64
    )
    device = args.device
    model = QueryAgentR1(config).to(device)
    dataloader = get_dataloader(batch_size=args.batch_size)

    print("=== Stage 1: SFT ===")
    sft_train(model, dataloader, epochs=args.sft_epochs, device=device)

    print("\n=== Stage 2: GRPO RL ===")
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    rl_train(model, ref_model, dataloader, epochs=args.rl_epochs, device=device)

    torch.save(model.state_dict(), "queryagent_r1.pt")
    print("Saved model to queryagent_r1.pt")


if __name__ == "__main__":
    main()
