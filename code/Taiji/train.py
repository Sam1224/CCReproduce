"""
Taiji Training Pipeline: two-stage (SFT → RL with POPO).

Stage 1 (SFT): Fine-tune LLM on reverse-engineered CoT data
Stage 2 (RL):  POPO training to align semantic and ID rewards
"""
import argparse
import json
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.cot_data_gen import generate_sft_dataset
from model.popo import POPO, POPOConfig, SemanticReward, IDReward, RejectionSampler


class CoTDataset(Dataset):
    """Dataset for SFT training on CoT data."""

    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        with open(data_path) as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = sample["prompt"] + " " + sample["completion"]
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_encoding = self.tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = encoding["input_ids"].squeeze().clone()
        labels[:prompt_len] = -100  # mask prompt in loss

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def train_sft(args, llm_enhancer):
    """Stage 1: Supervised Fine-Tuning on reverse-engineered CoT data."""
    print("\n=== Stage 1: SFT Training ===")

    # Generate SFT data if not exists
    sft_path = "data/toy_data/sft_train.json"
    if not Path(sft_path).exists():
        generate_sft_dataset("data/toy_data/train.json", sft_path)

    dataset = CoTDataset(sft_path, llm_enhancer.tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(llm_enhancer.llm.parameters(), lr=args.sft_lr)

    llm_enhancer.train()
    for epoch in range(args.sft_epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"SFT Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(llm_enhancer.device)
            attention_mask = batch["attention_mask"].to(llm_enhancer.device)
            labels = batch["labels"].to(llm_enhancer.device)

            outputs = llm_enhancer.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm_enhancer.llm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    # Save SFT checkpoint
    ckpt_dir = Path(args.output_dir) / "sft_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    llm_enhancer.llm.save_pretrained(str(ckpt_dir))
    llm_enhancer.tokenizer.save_pretrained(str(ckpt_dir))
    print(f"SFT checkpoint saved to {ckpt_dir}")


def train_rl_popo(args, llm_enhancer, online_ranker):
    """
    Stage 2: RL training with POPO.

    Each RL step:
    1. Sample a user batch
    2. Generate intent embeddings from LLM (current policy)
    3. Score with semantic and ID rewards
    4. Compute POPO loss with Pareto-optimal weights
    5. Update LLM parameters
    6. Update Pareto weights from gradients
    """
    print("\n=== Stage 2: RL Training with POPO ===")

    with open("data/toy_data/train.json") as f:
        users = json.load(f)

    popo_config = POPOConfig(
        lr=args.rl_lr,
        clip_eps=0.2,
        kl_coef=0.01,
    )
    popo = POPO(popo_config)
    semantic_reward = SemanticReward(embed_dim=128).to(llm_enhancer.device)
    id_reward = IDReward()
    rejection_sampler = RejectionSampler()

    # Item ID embeddings (random init for toy, would be pretrained in production)
    n_items = 500
    id_embed_table = torch.nn.Embedding(n_items, 64).to(llm_enhancer.device)
    optimizer = optim.AdamW(
        list(llm_enhancer.llm.parameters()) + list(id_embed_table.parameters()),
        lr=args.rl_lr,
    )

    old_log_probs = None

    for step in tqdm(range(args.rl_steps), desc="RL Steps"):
        # Sample a mini-batch of users
        import random
        batch_users = random.sample(users, min(args.batch_size, len(users)))

        # Forward pass: get intent embeddings
        prompts = [llm_enhancer.format_user_sequence(u) for u in batch_users]
        inputs = llm_enhancer.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(llm_enhancer.device)

        outputs = llm_enhancer.llm(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].mean(dim=1)
        intent_embeddings = llm_enhancer.intent_projector(hidden)  # [B, 128]

        # Reference embeddings (would be from a pretrained encoder in production)
        ref_embeddings = torch.randn_like(intent_embeddings)
        r_sem = semantic_reward(intent_embeddings, ref_embeddings)  # [B]

        # ID rewards: sample random items for each user and predict CTR/CVR
        item_ids = torch.randint(0, n_items, (len(batch_users),)).to(llm_enhancer.device)
        item_embeds = id_embed_table(item_ids)  # [B, 64]
        ranker_out = online_ranker(intent_embeddings.detach(), item_embeds)
        r_id = id_reward(ranker_out["ctr"], ranker_out["cvr"])  # [B]

        # Log probabilities from LLM (simplified: use mean logit as proxy)
        log_probs = outputs.logits.mean(dim=[1, 2])  # [B] (proxy)

        if old_log_probs is None:
            old_log_probs = log_probs.detach()

        # POPO loss
        popo_out = popo.popo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            semantic_rewards=r_sem,
            id_rewards=r_id,
        )
        loss = popo_out["loss"]

        optimizer.zero_grad()
        loss.backward()

        # Collect gradients for Pareto weight update
        all_grads = torch.cat([
            p.grad.flatten() for p in llm_enhancer.llm.parameters()
            if p.grad is not None
        ])
        # In production: compute separate gradients per objective for POPO update
        # Here: approximate with noise perturbation (pseudocode placeholder)
        grad_sem = all_grads + 0.01 * torch.randn_like(all_grads)  # proxy
        grad_id = all_grads - 0.01 * torch.randn_like(all_grads)   # proxy
        new_lam = popo.update_weight(grad_sem, grad_id)

        torch.nn.utils.clip_grad_norm_(llm_enhancer.llm.parameters(), 1.0)
        optimizer.step()

        old_log_probs = log_probs.detach()

        if step % 50 == 0:
            print(
                f"  Step {step}: loss={loss.item():.4f}, "
                f"r_sem={r_sem.mean().item():.4f}, r_id={r_id.mean().item():.4f}, "
                f"λ(sem)={new_lam:.3f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_lr", type=float, default=2e-5)
    parser.add_argument("--rl_steps", type=int, default=200)
    parser.add_argument("--rl_lr", type=float, default=1e-6)
    parser.add_argument("--output_dir", default="checkpoints/taiji")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Use tiny model for toy reproduction")
    args = parser.parse_args()

    print("Loading Taiji models...")
    # Use a tiny model for toy reproduction
    from model.llm_enhancer import TaijiLLMEnhancer, TaijiOnlineRanker
    llm_enhancer = TaijiLLMEnhancer(model_name=args.model_name)
    online_ranker = TaijiOnlineRanker().to(llm_enhancer.device)

    if args.stage in ("sft", "both"):
        train_sft(args, llm_enhancer)

    if args.stage in ("rl", "both"):
        train_rl_popo(args, llm_enhancer, online_ranker)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
