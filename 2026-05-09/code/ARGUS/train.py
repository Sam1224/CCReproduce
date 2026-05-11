"""
ARGUS — Training script
Three-stage training pipeline from arXiv 2605.02200.

Usage:
    python train.py --stage 1 --policy_version v2_edu_anxiety --epochs 3
    python train.py --stage 2 --policy_version v2_edu_anxiety --epochs 2
    python train.py --stage 3 --policy_version v2_edu_anxiety --epochs 2
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import AdGovernanceDataset, PolicyKnowledgeBase
from model import PolicyVLM, PDUArchitecture, HardSampleMiner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage I: Policy Seeding
# ---------------------------------------------------------------------------

def train_stage1(
    model: PolicyVLM,
    dataset: AdGovernanceDataset,
    policy_kb: PolicyKnowledgeBase,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 16,
    device: str = "cpu",
) -> PolicyVLM:
    """
    Stage I: Policy Seeding.

    Fine-tune VLM on new policy data (few-shot) to build initial perception.
    Paper: "Policy Seeding for initial perception"

    In production: use actual multimodal fine-tuning with LoRA.
    """
    logger.info("=== Stage I: Policy Seeding ===")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            images = batch["image"].to(device)
            texts = batch["text"]
            labels = batch["label"].to(device)

            # Augment text with policy context (RAG)
            augmented_texts = []
            for text in texts:
                retrieved = policy_kb.retrieve(text, top_k=2)
                policy_ctx = policy_kb.format_for_prompt(retrieved)
                augmented_texts.append(f"{text}\n\n{policy_ctx}")

            optimizer.zero_grad()
            out = model(images, augmented_texts)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (out["logits"].argmax(-1) == labels).sum().item()
            total += len(labels)

        acc = correct / total
        logger.info(f"  Stage I Epoch {epoch+1}/{epochs} — Loss={total_loss/total:.4f} Acc={acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# Stage II: Adversarial Label Rectification with RL
# ---------------------------------------------------------------------------

def train_stage2_rl(
    model: PolicyVLM,
    dataset: AdGovernanceDataset,
    policy_kb: PolicyKnowledgeBase,
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cpu",
) -> PolicyVLM:
    """
    Stage II: Adversarial Label Rectification via RL.

    PDU generates rectified labels → RL reward shapes model behavior.
    Paper: "Adversarial Label Rectification, which utilizes a
    'Prosecutor-Defender-Umpire' architecture to resolve conflicts"

    Simplified REINFORCE implementation; production uses PPO/GRPO.
    """
    logger.info("=== Stage II: Adversarial Label Rectification ===")
    pdu = PDUArchitecture(model, policy_kb)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_reward, total_samples = 0.0, 0

        for i, sample in enumerate(dataset):
            if i >= len(dataset) // 2:  # half dataset per epoch for speed
                break

            image = sample["image"].to(device)
            text = sample["text"]

            # PDU rectification pass (no_grad for agents)
            with torch.no_grad():
                pdu_result = pdu.rectify(image, text)

            # Forward pass to get model logits
            out = model(image.unsqueeze(0), [text])
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(-1).item()

            # RL reward from umpire verdict
            reward = pdu.compute_rl_reward(pdu_result, pred)
            total_reward += reward
            total_samples += 1

            # REINFORCE loss: -reward * log_prob(pred)
            log_prob = torch.log(probs[0, pred] + 1e-8)
            rl_loss = -reward * log_prob

            optimizer.zero_grad()
            rl_loss.backward()
            optimizer.step()

        avg_reward = total_reward / max(total_samples, 1)
        logger.info(
            f"  Stage II Epoch {epoch+1}/{epochs} — "
            f"Avg RL Reward={avg_reward:.4f} Samples={total_samples}"
        )

    return model


# ---------------------------------------------------------------------------
# Stage III: Latent Knowledge Discovery
# ---------------------------------------------------------------------------

def train_stage3(
    model: PolicyVLM,
    dataset: AdGovernanceDataset,
    policy_kb: PolicyKnowledgeBase,
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 16,
    device: str = "cpu",
) -> PolicyVLM:
    """
    Stage III: Latent Knowledge Discovery.

    Mine hard/gray-area samples, synthesize adversarial variants,
    and fine-tune model on expanded hard-negative dataset.
    Paper: "employs a tripartite dialectical discussion to unearth
    sophisticated, 'gray-area' violations"
    """
    logger.info("=== Stage III: Latent Knowledge Discovery ===")
    pdu = PDUArchitecture(model, policy_kb)
    miner = HardSampleMiner(pdu)

    # Mine hard samples
    logger.info("  Mining hard / gray-area samples...")
    hard_samples = miner.mine_hard_samples(dataset, model, max_samples=200)
    logger.info(f"  Found {len(hard_samples)} hard samples")

    if not hard_samples:
        logger.warning("  No hard samples found — skipping Stage III fine-tuning")
        return model

    # Fine-tune on hard samples
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for hs in hard_samples:
            # Use adversarial text + umpire label for training
            text = hs["adversarial_text"]
            label = torch.tensor(hs["umpire_label"], dtype=torch.long).to(device)
            # Toy: synthetic image
            image = torch.randn(1, 3, 224, 224).to(device)

            optimizer.zero_grad()
            out = model(image, [text])
            loss = criterion(out["logits"], label.unsqueeze(0))
            # Weight by umpire confidence
            loss = loss * hs["confidence"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out["logits"].argmax(-1).item()
            correct += (pred == hs["umpire_label"])
            total += 1

        acc = correct / max(total, 1)
        logger.info(
            f"  Stage III Epoch {epoch+1}/{epochs} — "
            f"Loss={total_loss/max(total,1):.4f} Acc={acc:.4f}"
        )

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ARGUS three-stage training")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--policy_version", default="v2_edu_anxiety",
                   choices=["v1_baseline", "v2_edu_anxiety", "v3_appearance_anxiety"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=400)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save_dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.save_dir).mkdir(exist_ok=True)

    logger.info(f"ARGUS — Stage {args.stage} training | Policy: {args.policy_version}")

    # Data
    dataset = AdGovernanceDataset(
        policy_version=args.policy_version,
        split="train",
        num_samples=args.num_samples,
    )
    policy_kb = PolicyKnowledgeBase(policy_version=args.policy_version)

    # Model
    model = PolicyVLM(device=args.device)

    if args.stage == 1:
        model = train_stage1(
            model, dataset, policy_kb,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
    elif args.stage == 2:
        model = train_stage2_rl(
            model, dataset, policy_kb,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
    elif args.stage == 3:
        model = train_stage3(
            model, dataset, policy_kb,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )

    # Save
    ckpt_path = Path(args.save_dir) / f"argus_stage{args.stage}_{args.policy_version}.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
