"""
Valley3 Stage 3: E-commerce Domain Knowledge Injection.

Goal: Inject e-commerce-specific knowledge into the Omni MLLM:
  - Product attribute extraction (属性抽取)
  - Compliance/violation detection (违规内容检测: 虚假宣传、违禁品)
  - Cross-border product description generation (跨境商品描述)
  - Short-video e-commerce content understanding (短视频电商理解)
  - Influencer/KOL content quality assessment (达人内容质量评估)

Paper key contribution: This stage is where Valley3 diverges from general-purpose
MLLMs by being trained on massive-scale proprietary e-commerce data from
Alibaba International Digital Commerce.

Training: Full model fine-tuning on e-commerce instruction-following data.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.valley3 import Valley3Model


# E-commerce task-specific loss weighting
TASK_WEIGHTS = {
    "product_attribute_extraction": 1.0,
    "violation_detection": 2.0,        # Higher weight: critical for platform safety
    "description_generation": 1.0,
    "short_video_understanding": 1.5,  # Key for Omni model differentiation
    "influencer_quality_assessment": 1.5,
}


def train_stage3(
    model: Valley3Model,
    dataloader,
    num_epochs: int = 2,
    lr: float = 5e-5,
    device: str = "cpu",
) -> Valley3Model:
    """
    Stage 3: E-commerce domain knowledge injection.
    Full model fine-tuning with task-weighted loss.
    """
    model = model.to(device)

    # All parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    print(f"Stage 3: Full model fine-tuning ({sum(p.numel() for p in model.parameters()):,} params)")
    print("  E-commerce tasks: attribute extraction, violation detection, short-video, influencer")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        task_losses = {t: 0.0 for t in TASK_WEIGHTS}
        task_counts = {t: 0 for t in TASK_WEIGHTS}

        for step, batch in enumerate(dataloader):
            task = batch.get("task", "product_attribute_extraction")
            batch_tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(
                input_ids=batch_tensors["input_ids"],
                pixel_values=batch_tensors.get("pixel_values"),
                mel_spectrograms=batch_tensors.get("mel_spectrograms"),
                labels=batch_tensors.get("labels"),
            )

            loss = outputs["loss"]
            if loss is None:
                continue

            # Apply task weight
            weight = TASK_WEIGHTS.get(str(task), 1.0)
            weighted_loss = loss * weight

            optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  Stage3 Epoch {epoch+1}/{num_epochs} Step {step}: "
                      f"loss={loss.item():.4f} (weight={weight:.1f})")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Stage 3 Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return model
