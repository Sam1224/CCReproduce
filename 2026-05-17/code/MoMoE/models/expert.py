"""
Predict Operator (Expert Models) for MoMoE.

Each expert is a small language model fine-tuned on a specific
community or norm-violation category.

Paper: "The Predict operator uses fine-tuned small language model experts,
like Llama or Mistral, specialized either by community (MoMoE-Community)
or norm violation category (MoMoE-NormVio)."
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ExpertPrediction:
    label: int          # 0 = not violation, 1 = violation
    confidence: float   # Probability of label=1
    expert_id: str      # Which expert made this prediction


class ContentModerationExpert(nn.Module):
    """
    A single community/norm-violation specialized expert.

    Each expert is a lightweight SLM (e.g., Llama-3.2-1B or DistilBERT)
    fine-tuned on data from a specific community or violation type.

    For reproduction: uses a distilbert-base-uncased as a lightweight
    stand-in for Llama/Mistral experts.
    """

    def __init__(
        self,
        expert_id: str,
        model_name: str = "distilbert-base-uncased",
        checkpoint: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.expert_id = expert_id
        self.device = device

        # In the paper: Llama-3.2-1B or Mistral-7B fine-tuned per community
        # For reproduction efficiency: DistilBERT (66M params)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name if checkpoint is None else checkpoint,
            num_labels=2,
        ).to(device)

    def forward(self, texts: list[str]) -> list[ExpertPrediction]:
        """Predict violation probability for each text."""
        self.model.eval()
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=-1)

        predictions = []
        for i in range(len(texts)):
            confidence = probs[i, 1].item()  # P(violation)
            label = 1 if confidence >= 0.5 else 0
            predictions.append(ExpertPrediction(
                label=label,
                confidence=confidence,
                expert_id=self.expert_id,
            ))
        return predictions

    def fine_tune_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One gradient step for fine-tuning this expert."""
        self.model.train()
        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }
        outputs = self.model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()


class ExpertPool:
    """
    Pool of specialized expert models.
    Manages loading, caching, and inference for multiple experts.
    """

    def __init__(
        self,
        expert_ids: list[str],
        model_name: str = "distilbert-base-uncased",
        checkpoint_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.expert_ids = expert_ids
        self.experts: dict[str, ContentModerationExpert] = {}

        for eid in expert_ids:
            ckpt = None
            if checkpoint_dir:
                ckpt_path = os.path.join(checkpoint_dir, f"expert_{eid}")
                if os.path.exists(ckpt_path):
                    ckpt = ckpt_path

            self.experts[eid] = ContentModerationExpert(
                expert_id=eid,
                model_name=model_name,
                checkpoint=ckpt,
                device=device,
            )

    def predict(
        self,
        texts: list[str],
        active_experts: list[str],
    ) -> dict[str, list[ExpertPrediction]]:
        """
        Get predictions from specified active experts.
        Returns dict mapping expert_id -> predictions.
        """
        results = {}
        for eid in active_experts:
            if eid in self.experts:
                results[eid] = self.experts[eid](texts)
        return results
