"""
Valley3 Stage 4: Long-Context Reasoning + Controllable Thinking.

Goal: Enable Valley3 to handle long-context e-commerce reasoning tasks:
  - Multi-turn product research sessions
  - Competitive analysis across multiple products
  - Deep compliance review with chain-of-thought
  - Agentic search trajectory learning (tool-use sequences)

Key paper contribution: Post-training for "controllable reasoning modes" —
the model can switch between:
  1. Non-thinking mode: fast response for real-time moderation
  2. Light thinking: moderate CoT for product attribute analysis
  3. Heavy thinking: deep CoT for complex compliance decisions

Training approach:
  - SFT on long-context e-commerce data
  - Preference optimization (DPO-style) for thinking mode calibration
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.valley3 import Valley3Model


# Special tokens for reasoning mode control (paper Section 4.3)
THINKING_MODE_TOKENS = {
    "non_thinking": "<|non_thinking|>",
    "light_thinking": "<|light_thinking|>",
    "heavy_thinking": "<|heavy_thinking|>",
}


def compute_dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss for preference optimization.
    Used to calibrate thinking-mode quality.

    Formula (Rafailov et al. 2023):
      L_DPO = -E[log σ(β(log π_θ(y_w|x) - log π_ref(y_w|x))
                      - β(log π_θ(y_l|x) - log π_ref(y_l|x)))]

    Here we use it to prefer "heavy thinking" responses for complex tasks.
    """
    log_ratios = policy_logps - reference_logps
    loss = -torch.nn.functional.logsigmoid(beta * log_ratios)
    return loss.mean()


def train_stage4(
    model: Valley3Model,
    dataloader,
    num_epochs: int = 1,
    lr: float = 2e-5,
    device: str = "cpu",
) -> Valley3Model:
    """
    Stage 4: Long-context reasoning + controllable thinking post-training.
    Lower learning rate to preserve Stage 3 domain knowledge.
    """
    model = model.to(device)

    # All parameters trainable but with lower LR
    for param in model.parameters():
        param.requires_grad = True

    print(f"Stage 4: Long-context + controllable thinking training")
    print(f"  LR={lr} (lower to preserve e-commerce domain knowledge)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            batch_tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Randomly sample reasoning mode for this batch (curriculum)
            mode = torch.randint(0, 3, (1,)).item()
            mode_map = {0: "non_thinking", 1: "light_thinking", 2: "heavy_thinking"}
            reasoning_mode = mode_map[mode]

            outputs = model(
                input_ids=batch_tensors["input_ids"],
                pixel_values=batch_tensors.get("pixel_values"),
                mel_spectrograms=batch_tensors.get("mel_spectrograms"),
                labels=batch_tensors.get("labels"),
                reasoning_mode=reasoning_mode,
            )

            loss = outputs["loss"]
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  Stage4 Epoch {epoch+1}/{num_epochs} Step {step}: "
                      f"loss={loss.item():.4f} mode={reasoning_mode}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Stage 4 Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return model
