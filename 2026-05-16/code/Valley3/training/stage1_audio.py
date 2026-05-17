"""
Stage 1: Audio Understanding Pre-training (Valley3 §3.1)

Goal: Align the multilingual audio encoder to the LLM embedding space.
Only the audio encoder and audio projector are trained; LLM backbone is frozen.

Valley3 paper: "Valley3 progressively acquires audio understanding through
a four-stage omni e-commerce continued pre-training pipeline.
Stage 1 focuses on audio representation alignment with audio-text pairs
from multilingual e-commerce livestreams."
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.valley3 import Valley3Model, Valley3Config
from data.toy_ecom_dataset import ToyEcomDataset, collate_fn


def stage1_train(
    cfg: Valley3Config,
    num_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cpu",
):
    """
    Stage 1 training: audio encoder + audio projector only.
    LLM backbone and vision encoder are frozen.
    """
    model = Valley3Model(cfg).to(device)

    # Freeze everything except audio components
    for name, param in model.named_parameters():
        if "audio" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 — Trainable: {trainable:,} / {total:,} params")

    dataset = ToyEcomDataset(stage=1, num_samples=64, seq_len=128, audio_frames=1500)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            mel_features = batch["mel_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                mel_features=mel_features,
                labels=labels,
            )
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 5 == 0:
                print(f"  Epoch {epoch+1} Step {step+1}/{len(loader)} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}")

    return model


if __name__ == "__main__":
    cfg = Valley3Config(llm_hidden_size=512, num_heads=8, num_layers=2)
    model = stage1_train(cfg, num_epochs=2, batch_size=2)
    print("Stage 1 pre-training complete.")
