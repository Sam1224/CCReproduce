"""
Allocate Operator for MoMoE.

The Allocator determines which experts to invoke and their weights.
Implementation: RoBERTa-base fine-tuned for community/norm-violation classification.

Paper: "A RoBERTa-base model is fine-tuned on either D_Community
(to predict source subreddit, 7 classes) or D_NormVio (to predict
norm violation category, 5 classes), with softmax scores from the
logits serving as allocation weights for the corresponding experts."
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional


COMMUNITY_LABELS = [
    "askreddit", "relationships", "gaming", "worldnews",
    "technology", "science", "fitness"
]

NORM_VIOLATION_LABELS = [
    "harassment", "hate_speech", "misinformation",
    "spam_scam", "explicit_content"
]


@dataclass
class AllocationResult:
    weights: torch.Tensor       # Shape: (n_experts,), softmax scores
    top_k_indices: list[int]    # Top-K expert indices
    top_k_weights: list[float]  # Corresponding weights
    expert_labels: list[str]    # Expert names


class Allocator(nn.Module):
    """
    RoBERTa-base classifier for expert allocation.

    Fine-tuned as a standard sequence classifier; softmax probabilities
    serve directly as expert weights for the Aggregate step.
    """

    def __init__(
        self,
        allocator_type: str = "community",  # "community" or "norm_violation"
        model_name: str = "roberta-base",
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.allocator_type = allocator_type
        self.labels = COMMUNITY_LABELS if allocator_type == "community" else NORM_VIOLATION_LABELS
        self.n_experts = len(self.labels)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name if checkpoint is None else checkpoint,
            num_labels=self.n_experts,
        )

    def forward(
        self,
        texts: list[str],
        top_k: int = 3,
        device: str = "cpu",
    ) -> list[AllocationResult]:
        """
        Allocate experts for a batch of texts.

        Args:
            texts: Input posts/comments
            top_k: Number of top experts to activate
        Returns:
            List of AllocationResult, one per input
        """
        self.model.to(device)
        self.model.eval()

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = self.model(**encodings).logits  # (B, n_experts)
            weights = torch.softmax(logits, dim=-1)  # Allocation weights

        results = []
        for i in range(len(texts)):
            w = weights[i]
            top_k_vals, top_k_idx = torch.topk(w, k=min(top_k, self.n_experts))
            results.append(AllocationResult(
                weights=w,
                top_k_indices=top_k_idx.tolist(),
                top_k_weights=top_k_vals.tolist(),
                expert_labels=[self.labels[j] for j in top_k_idx.tolist()],
            ))

        return results

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer) -> float:
        """Single training step for fine-tuning the allocator."""
        self.model.train()
        labels = batch["labels"].to(next(self.model.parameters()).device)
        encodings = {k: v.to(labels.device) for k, v in batch["encodings"].items()}

        outputs = self.model(**encodings, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
