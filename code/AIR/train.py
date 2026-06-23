"""
AIR Training Pipeline: two-stage (LLM SFT → Knowledge Distillation).

Stage 1 (SFT): Fine-tune LLM reasoner on (AIU, cross-domain intent) pairs
Stage 2 (KD):  Distill LLM teacher into lightweight online student model
"""
import argparse
import json
import random
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.toy_dataset import build_dataset
from model.aiu_extractor import AIUExtractorRuleBased, AIUExtractorNeural, CONTENT_CATEGORIES
from model.llm_reasoner import AIRLLMReasoner
from model.knowledge_distill import AIRStudentModel, KnowledgeDistillationTrainer, AIROnlineRanker


CREATOR_TIERS = ["nano", "micro", "macro", "mega"]


class AIRSFTDataset(Dataset):
    """SFT dataset: (AIU text, target reasoning) pairs."""

    def __init__(self, data_path: str):
        with open(data_path) as f:
            self.users = json.load(f)
        self.extractor = AIUExtractorRuleBased()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        aiu_text = self.extractor.extract(user)
        # Use the pre-generated AIU label as completion target
        label = user.get("aiu_label", "")
        return {"aiu_text": aiu_text, "label": label}


class AIRKDDataset(Dataset):
    """KD dataset: (content_history, user) pairs for student training."""

    def __init__(self, data_path: str, max_seq_len: int = 25):
        with open(data_path) as f:
            self.users = json.load(f)
        self.max_seq_len = max_seq_len
        self.cat2id = {c: i for i, c in enumerate(CONTENT_CATEGORIES)}
        self.tier2id = {t: i for i, t in enumerate(CREATOR_TIERS)}

    def __len__(self):
        return len(self.users)

    def _encode_history(self, content_history: list) -> tuple:
        seq = content_history[:self.max_seq_len]
        T = self.max_seq_len
        cat_ids = torch.zeros(T, dtype=torch.long)
        tier_ids = torch.zeros(T, dtype=torch.long)
        signals = torch.zeros(T, 3)
        mask = torch.ones(T, dtype=torch.bool)  # True = padding

        for i, h in enumerate(seq):
            cat_ids[i] = self.cat2id.get(h["category"], 0)
            tier_ids[i] = self.tier2id.get(h["creator_tier"], 0)
            signals[i] = torch.tensor([
                float(h.get("watched", 0)),
                float(h.get("liked", 0)),
                float(h.get("followed_creator", 0)),
            ])
            mask[i] = False  # not padding

        return cat_ids, tier_ids, signals, mask

    def __getitem__(self, idx):
        user = self.users[idx]
        cat_ids, tier_ids, signals, mask = self._encode_history(user["content_history"])
        # Product category label (first purchase category, if any)
        from data.toy_dataset import PRODUCT_CATEGORIES
        purchases = user.get("purchase_history", [])
        if purchases:
            cat_label = PRODUCT_CATEGORIES.index(purchases[0]["category"]) if purchases[0]["category"] in PRODUCT_CATEGORIES else 0
        else:
            cat_label = 0
        return {
            "cat_ids": cat_ids,
            "tier_ids": tier_ids,
            "signals": signals,
            "padding_mask": mask,
            "cat_label": torch.tensor(cat_label, dtype=torch.long),
            "aiu_text": AIUExtractorRuleBased().extract(user),
        }


def train_sft(args, llm_reasoner: AIRLLMReasoner):
    """Stage 1: SFT on (AIU text, cross-domain reasoning) pairs."""
    print("\n=== Stage 1: AIR SFT Training ===")

    dataset = AIRSFTDataset(f"data/toy_data/train.json")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: b)

    optimizer = optim.AdamW(llm_reasoner.llm.parameters(), lr=args.sft_lr)

    llm_reasoner.train()
    for epoch in range(args.sft_epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"AIR SFT Epoch {epoch+1}"):
            aiu_texts = [s["aiu_text"] for s in batch]
            labels = [s["label"] for s in batch]

            out = llm_reasoner(aiu_texts=aiu_texts, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm_reasoner.llm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}: avg_loss={total_loss/len(loader):.4f}")

    ckpt_dir = Path(args.output_dir) / "sft_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    llm_reasoner.llm.save_pretrained(str(ckpt_dir))
    llm_reasoner.tokenizer.save_pretrained(str(ckpt_dir))
    print(f"SFT checkpoint saved to {ckpt_dir}")


def train_kd(args, llm_reasoner: AIRLLMReasoner):
    """Stage 2: Knowledge Distillation from LLM teacher to lightweight student."""
    print("\n=== Stage 2: AIR Knowledge Distillation ===")

    dataset = AIRKDDataset("data/toy_data/train.json")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = llm_reasoner.device

    aiu_extractor = AIUExtractorNeural(
        n_content_cats=10,
        n_creator_tiers=4,
        hidden_dim=128,
        aiu_dim=256,
    ).to(device)

    student = AIRStudentModel(
        aiu_dim=256,
        intent_dim=256,
        n_product_cats=10,
    ).to(device)

    kd_trainer = KnowledgeDistillationTrainer(
        student=student,
        distill_weight=1.0,
        category_weight=0.3,
        contrastive_weight=0.2,
    ).to(device)

    online_ranker = AIROnlineRanker(intent_dim=256, product_dim=64).to(device)

    optimizer = optim.AdamW(
        list(aiu_extractor.parameters()) + list(student.parameters()),
        lr=args.kd_lr,
    )

    llm_reasoner.eval()

    for epoch in range(args.kd_epochs):
        total_loss = 0.0
        aiu_extractor.train()
        student.train()

        for batch in tqdm(loader, desc=f"KD Epoch {epoch+1}"):
            cat_ids = batch["cat_ids"].to(device)
            tier_ids = batch["tier_ids"].to(device)
            signals = batch["signals"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            cat_labels = batch["cat_label"].to(device)
            aiu_texts = batch["aiu_text"]

            # Student forward: neural AIU → AIU embedding
            aiu_emb = aiu_extractor(cat_ids, tier_ids, signals, padding_mask)  # [B, 256]

            # Teacher forward: LLM text → teacher intent embedding (no grad)
            with torch.no_grad():
                teacher_out = llm_reasoner(aiu_texts=list(aiu_texts))
                teacher_intent = teacher_out["intent_embeddings"]  # [B, 256]

            # KD loss
            kd_out = kd_trainer(
                aiu_embeddings=aiu_emb,
                teacher_intent=teacher_intent,
                product_cat_labels=cat_labels,
            )
            loss = kd_out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(aiu_extractor.parameters()) + list(student.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"  KD Epoch {epoch+1}: avg_loss={avg:.4f}")

    # Save KD checkpoint
    ckpt_dir = Path(args.output_dir) / "kd_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "aiu_extractor": aiu_extractor.state_dict(),
        "student": student.state_dict(),
        "online_ranker": online_ranker.state_dict(),
    }, str(ckpt_dir / "model.pt"))
    print(f"KD checkpoint saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sft", "kd", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--sft_lr", type=float, default=2e-5)
    parser.add_argument("--kd_epochs", type=int, default=3)
    parser.add_argument("--kd_lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="checkpoints/air")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # Generate data if needed
    if not Path("data/toy_data/train.json").exists():
        print("Generating toy dataset...")
        build_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AIR LLM Reasoner on {device}...")
    llm_reasoner = AIRLLMReasoner(model_name=args.model_name, device=device)

    if args.stage in ("sft", "both"):
        train_sft(args, llm_reasoner)

    if args.stage in ("kd", "both"):
        train_kd(args, llm_reasoner)

    print("\nAIR training complete.")


if __name__ == "__main__":
    main()
