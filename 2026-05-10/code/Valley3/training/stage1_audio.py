"""
Valley3 Stage 1: Audio Understanding Pre-training.

Goal: Extend the vision-language model with native audio understanding,
specifically targeting multilingual e-commerce short-video speech (达人口播).

Training objective: Next-token prediction on audio transcription sequences.
Frozen: LLM backbone (only audio encoder + A-projector are trained).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict

from model.valley3 import Valley3Model, Valley3Config


def train_stage1(
    model: Valley3Model,
    dataloader,
    num_epochs: int = 1,
    lr: float = 2e-4,
    device: str = "cpu",
) -> Valley3Model:
    """
    Stage 1: Train audio encoder + A-projector only.
    The LLM backbone and vision components are frozen.

    Paper: Stage 1 focuses on audio understanding capability acquisition,
    enabling the model to process speech from short-video e-commerce content.
    """
    model = model.to(device)

    # Freeze all except audio components
    for name, param in model.named_parameters():
        if "audio_encoder" in name or "audio_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Stage 1: Training {sum(p.numel() for p in trainable_params):,} parameters")
    print("  (audio_encoder + audio_projector only, LLM backbone frozen)")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Only use audio + text (no pixel_values in Stage 1)
            outputs = model(
                input_ids=batch["input_ids"],
                mel_spectrograms=batch.get("mel_spectrograms"),
                labels=batch.get("labels"),
            )

            loss = outputs["loss"]
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  Stage1 Epoch {epoch+1}/{num_epochs} Step {step}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Stage 1 Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return model
