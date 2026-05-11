"""
Valley3 Stage 2: Cross-Modal Instruction Following.

Goal: Train the model on diverse multimodal instruction-following data,
enabling unified understanding of text, image, video, and audio.

Training objective: Instruction-following loss (next-token prediction
on response tokens only; instruction tokens masked with -100).
Trained: V-projector + A-projector (vision/audio encoders still frozen,
LLM backbone unfrozen for cross-modal alignment).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.valley3 import Valley3Model


def train_stage2(
    model: Valley3Model,
    dataloader,
    num_epochs: int = 1,
    lr: float = 1e-4,
    device: str = "cpu",
) -> Valley3Model:
    """
    Stage 2: Cross-modal instruction following.
    Unfreeze LLM backbone + projectors; keep encoders frozen.
    """
    model = model.to(device)

    # Unfreeze LLM layers + projectors
    for name, param in model.named_parameters():
        if any(x in name for x in ["layers", "vision_projector", "audio_projector", "norm", "lm_head", "token_embedding"]):
            param.requires_grad = True
        elif any(x in name for x in ["vision_encoder", "audio_encoder"]):
            param.requires_grad = False  # encoders still frozen
        else:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Stage 2: Training {sum(p.numel() for p in trainable_params):,} parameters")
    print("  (LLM layers + projectors; vision/audio encoders frozen)")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
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
                print(f"  Stage2 Epoch {epoch+1}/{num_epochs} Step {step}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Stage 2 Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return model
