"""
Training script for Thinking Broad, Acting Fast (arXiv: 2601.21611)
E-Commerce Relevance via Multi-Perspective CoT Distillation

Usage:
    python train.py --data_path data/ecom_relevance.jsonl --epochs 5 --batch_size 32
"""

import argparse
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from model import (
    TeacherLLM,
    StudentRelevanceModel,
    TeacherLatentExtractor,
    LatentReasoningDistillationLoss,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EComRelevanceDataset(Dataset):
    """
    E-commerce query-product relevance dataset.

    Expected JSONL format:
    {
        "query": "...",
        "product_title": "...",
        "product_attrs": "...",   # optional
        "label": 1,               # 1=relevant, 0=irrelevant
        "teacher_reasoning": {    # pre-generated teacher CoT (optional)
            "semantic_match": "...",
            "user_intent": "...",
            "attribute_match": "...",
            "long_tail_coverage": "..."
        }
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_query_len: int = 64,
        max_product_len: int = 128,
        use_precomputed_reasoning: bool = True,
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_product_len = max_product_len
        self.use_precomputed_reasoning = use_precomputed_reasoning

        if os.path.exists(data_path):
            with open(data_path) as f:
                for line in f:
                    ex = json.loads(line.strip())
                    self.examples.append(ex)
        else:
            # Toy data for demonstration
            self.examples = self._generate_toy_data()

    def _generate_toy_data(self) -> List[Dict]:
        """Generate toy examples for interface demonstration."""
        return [
            {
                "query": "running shoes for women",
                "product_title": "Nike Air Zoom Women's Running Shoes",
                "product_attrs": "color: blue, size: US 7, material: mesh",
                "label": 1,
                "teacher_reasoning": {
                    "semantic_match": "The product title 'Running Shoes' directly matches query 'running shoes'. Gender 'Women's' matches 'for women'. High semantic relevance.",
                    "user_intent": "User wants running shoes for female use. Product is specifically designed for women's running. Intent fully satisfied.",
                    "attribute_match": "No explicit size/color constraints in query. Product attributes (mesh, air zoom) suggest running-appropriate features.",
                    "long_tail_coverage": "Query is not particularly long-tail. Standard product match.",
                },
            },
            {
                "query": "bamboo fiber organic baby onesie",
                "product_title": "Cotton Baby Bodysuit 3-Pack",
                "product_attrs": "material: cotton, size: 0-3 months",
                "label": 0,
                "teacher_reasoning": {
                    "semantic_match": "Query specifies 'bamboo fiber organic' but product is 'cotton'. Material mismatch causes low semantic relevance.",
                    "user_intent": "User wants eco-friendly organic baby clothing. Cotton product may not satisfy the 'organic bamboo' intent.",
                    "attribute_match": "Critical attribute mismatch: 'bamboo fiber' vs 'cotton'. Query constraint not satisfied.",
                    "long_tail_coverage": "This is a long-tail query specifying both material (bamboo) and property (organic). Product fails to cover these niche requirements.",
                },
            },
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # In practice, tokenize query and product
        # Here: return dummy tensors sized for the interface
        query_input_ids = torch.zeros(self.max_query_len, dtype=torch.long)
        query_attention_mask = torch.ones(self.max_query_len, dtype=torch.long)
        product_input_ids = torch.zeros(self.max_product_len, dtype=torch.long)
        product_attention_mask = torch.ones(self.max_product_len, dtype=torch.long)

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "product_input_ids": product_input_ids,
            "product_attention_mask": product_attention_mask,
            "label": torch.tensor(ex["label"], dtype=torch.float),
            "teacher_reasoning": ex.get("teacher_reasoning", {}),
            "query": ex["query"],
            "product_title": ex["product_title"],
        }


# ---------------------------------------------------------------------------
# Two-Stage Training Pipeline
# ---------------------------------------------------------------------------

def stage1_generate_teacher_reasoning(
    dataset_path: str,
    teacher_llm: TeacherLLM,
    output_path: str,
):
    """
    Stage 1 (offline): Generate multi-perspective CoT reasoning from teacher LLM.
    Results are cached for Stage 2 distillation training.
    """
    examples = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            reasoning = teacher_llm.generate_multi_perspective_reasoning(
                query=ex["query"],
                product_title=ex["product_title"],
                product_attrs=ex.get("product_attrs", ""),
            )
            ex["teacher_reasoning"] = reasoning
            examples.append(ex)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[Stage 1] Teacher reasoning saved to {output_path}")


def stage2_distillation_training(
    args: argparse.Namespace,
    student_model: StudentRelevanceModel,
    teacher_extractor: TeacherLatentExtractor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
):
    """
    Stage 2: Distillation training of student model using teacher latent representations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_extractor = teacher_extractor.to(device)

    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = LatentReasoningDistillationLoss(lambda_distill=args.lambda_distill)

    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        total_rel_loss = 0.0
        total_align_loss = 0.0

        for batch in train_loader:
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            product_input_ids = batch["product_input_ids"].to(device)
            product_attention_mask = batch["product_attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Get teacher latent representations (batch-level)
            teacher_latents = []
            for i, reasoning in enumerate(batch["teacher_reasoning"]):
                with torch.no_grad():
                    t_latent = teacher_extractor(reasoning)
                teacher_latents.append(t_latent)
            teacher_latent = torch.cat(teacher_latents, dim=0).to(device)

            # Forward pass through student
            student_scores, student_latent = student_model(
                query_input_ids, query_attention_mask,
                product_input_ids, product_attention_mask,
            )

            # Compute distillation loss
            losses = criterion(student_scores, student_latent, teacher_latent, labels)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses["total"].item()
            total_rel_loss += losses["relevance"].item()
            total_align_loss += losses["alignment"].item()

        scheduler.step()
        n = len(train_loader)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {total_loss/n:.4f} | "
            f"Relevance: {total_rel_loss/n:.4f} | "
            f"Alignment: {total_align_loss/n:.4f}"
        )

        if val_loader is not None:
            eval_student(student_model, val_loader, device)

    return student_model


def eval_student(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            scores, _ = model(
                batch["query_input_ids"].to(device),
                batch["query_attention_mask"].to(device),
                batch["product_input_ids"].to(device),
                batch["product_attention_mask"].to(device),
            )
            preds = (scores > 0).long()
            labels = batch["label"].long().to(device)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"  Val Accuracy: {100*correct/total:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Thinking Broad Acting Fast model")
    parser.add_argument("--data_path", default="data/ecom_relevance.jsonl")
    parser.add_argument("--with_teacher_reasoning", default="data/ecom_relevance_with_cot.jsonl")
    parser.add_argument("--encoder_name", default="bert-base-uncased")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda_distill", type=float, default=0.5)
    parser.add_argument("--num_perspectives", type=int, default=4)
    parser.add_argument("--output_dir", default="checkpoints/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Thinking Broad, Acting Fast — Training Pipeline")
    print("arXiv: 2601.21611 | WWW 2026")
    print("=" * 60)

    # Stage 1: Generate teacher CoT reasoning (offline, expensive)
    teacher_llm = TeacherLLM()
    if not os.path.exists(args.with_teacher_reasoning):
        if os.path.exists(args.data_path):
            print("[Stage 1] Generating teacher multi-perspective CoT reasoning...")
            stage1_generate_teacher_reasoning(
                args.data_path, teacher_llm, args.with_teacher_reasoning
            )
        else:
            print("[Stage 1] No data found; using toy data for demonstration.")

    # Stage 2: Distillation training
    # Toy tokenizer placeholder
    tokenizer = None

    train_dataset = EComRelevanceDataset(
        args.with_teacher_reasoning if os.path.exists(args.with_teacher_reasoning) else args.data_path,
        tokenizer=tokenizer,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    student_model = StudentRelevanceModel(
        encoder_name=args.encoder_name,
        hidden_dim=args.hidden_dim,
        num_perspectives=args.num_perspectives,
    )
    teacher_extractor = TeacherLatentExtractor(hidden_dim=args.hidden_dim)

    print(f"\n[Stage 2] Distillation training for {args.epochs} epochs...")
    trained_model = stage2_distillation_training(
        args, student_model, teacher_extractor, train_loader
    )

    # Save
    torch.save(trained_model.state_dict(), os.path.join(args.output_dir, "student_model.pt"))
    print(f"\nModel saved to {args.output_dir}/student_model.pt")


if __name__ == "__main__":
    main()
