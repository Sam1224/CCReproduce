"""
Stage 3: E-commerce Domain Knowledge Pre-training (Valley3 §3.3)

Goal: Instill e-commerce domain knowledge across three core tasks:
  1. Product Understanding: title, attribute, category classification
  2. Livestream Analysis: host behavior, product demo quality, audience interaction
  3. Moderation & Governance: violation detection, policy adherence

Valley3 paper: "Stage 3 focuses on e-commerce domain knowledge acquisition.
The model learns to handle Product Understanding, Livestream Content Analysis,
and Moderation & Governance tasks, achieving an average absolute improvement
of over 7% compared to the baselines."

Training data (Valley3 full):
  - Product catalog: (image, title, category, attributes) tuples
  - Livestream clips: (video_frames, audio, transcript, QA annotation)
  - Violation corpus: (content, violation_type, policy_reference, rationale)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.valley3 import Valley3Model, Valley3Config
from data.toy_ecom_dataset import ToyEcomDataset, VIOLATION_TYPES, collate_fn


class EcomGovernanceHead(nn.Module):
    """
    Auxiliary classification head for violation detection (Stage 3 governance tasks).
    Valley3 uses generative output; this head is an auxiliary training signal.
    Formula: L_total = L_lm + λ * L_violation_cls (Valley3 §3.3)
    """
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state: [B, D] — pooled representation
        return self.classifier(hidden_state)


class Stage3EcomTrainer:
    """
    Stage 3 trainer: all parameters unfrozen, e-commerce domain data.
    Combines generative LM loss with auxiliary violation classification.
    """

    def __init__(self, cfg: Valley3Config, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.model = Valley3Model(cfg).to(device)
        self.governance_head = EcomGovernanceHead(
            cfg.llm_hidden_size, len(VIOLATION_TYPES)
        ).to(device)
        self.violation_loss_weight = 0.3  # λ in L_total formula

    def compute_violation_loss(
        self,
        logits: torch.Tensor,
        violation_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary violation classification loss.
        Uses the first token's hidden state as sequence representation.
        L_violation = CrossEntropy(governance_head(h_0), violation_label)
        """
        # Extract [CLS]-like representation from first position
        first_token_logits = logits[:, 0, :self.cfg.llm_hidden_size]
        # Project down for classification (simplified: use first D dims of vocab logits)
        hidden = logits[:, 0, :self.cfg.llm_hidden_size]
        cls_logits = self.governance_head(hidden.detach())
        mask = violation_labels >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return F.cross_entropy(cls_logits[mask], violation_labels[mask])

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        self.governance_head.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            pixel_values = batch.get("pixel_values")
            mel_features = batch.get("mel_features")
            violation_labels = batch.get("violation_label", torch.full((input_ids.size(0),), -1))

            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            if mel_features is not None:
                mel_features = mel_features.to(self.device)
            violation_labels = violation_labels.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mel_features=mel_features,
                labels=labels,
            )

            lm_loss = outputs["loss"]
            violation_loss = self.compute_violation_loss(outputs["logits"], violation_labels)

            # L_total = L_lm + λ * L_violation_cls
            loss = lm_loss + self.violation_loss_weight * violation_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.governance_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            if step % 5 == 0:
                print(
                    f"  Step {step+1} | LM: {lm_loss.item():.4f} "
                    f"| Violation: {violation_loss.item():.4f} "
                    f"| Total: {loss.item():.4f}"
                )

        return total_loss / len(loader)


def stage3_train(
    cfg: Valley3Config,
    num_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-5,
    device: str = "cpu",
):
    trainer = Stage3EcomTrainer(cfg, device)
    dataset = ToyEcomDataset(stage=3, num_samples=64, seq_len=256, image_size=224)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    all_params = list(trainer.model.parameters()) + list(trainer.governance_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(loader))

    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(loader, optimizer)
        scheduler.step()
        print(f"Stage 3 Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    return trainer


if __name__ == "__main__":
    cfg = Valley3Config(llm_hidden_size=512, num_heads=8, num_layers=2)
    trainer = stage3_train(cfg, num_epochs=2, batch_size=2)
    print("Stage 3 e-commerce pre-training complete.")
