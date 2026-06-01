#!/usr/bin/env python
"""Stage 1: Audio Understanding Pre-training (Valley3 toy)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.audio_encoder import AudioTransformerEncoder, AudioMLPConnector

def generate_toy_audio_data(n=200, T=64, mel_bins=80, d_model=128):
    """Toy: audio-text pairs where audio encodes a class label."""
    mels, texts = [], []
    for i in range(n):
        cls = i % 5
        mel = torch.randn(T, mel_bins) + cls * 0.5
        text_emb = torch.randn(d_model) + cls * 0.5
        mels.append(mel)
        texts.append(text_emb)
    return torch.stack(mels), torch.stack(texts)

def train(epochs=3, device="cpu"):
    print("Stage 1: Audio Understanding Pre-training")
    encoder = AudioTransformerEncoder(mel_bins=80, output_dim=128,
                                      d_model=128, nhead=4, num_layers=2).to(device)
    connector = AudioMLPConnector(audio_dim=128, llm_dim=128, num_tokens=4).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(connector.parameters()), lr=1e-3)
    mels, texts = generate_toy_audio_data()

    for epoch in range(epochs):
        perm = torch.randperm(len(mels))
        total_loss = 0.0
        for i in range(0, len(mels), 16):
            idx = perm[i:i+16]
            mel_b = mels[idx].to(device)
            text_b = texts[idx].to(device)

            audio_emb = encoder(mel_b)                    # [B, 128]
            audio_tokens = connector(audio_emb)           # [B, 4, 128]
            # Contrastive loss: audio embedding should be close to text embedding
            audio_pool = audio_tokens.mean(1)             # [B, 128]
            audio_norm = F.normalize(audio_pool, dim=-1)
            text_norm = F.normalize(text_b, dim=-1)
            sims = audio_norm @ text_norm.T               # [B, B]
            labels = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(sims * 10.0, labels) + \
                   F.cross_entropy(sims.T * 10.0, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss:.4f}")

    os.makedirs("data/ckpt", exist_ok=True)
    torch.save(encoder.state_dict(), "data/ckpt/audio_enc.pt")
    torch.save(connector.state_dict(), "data/ckpt/audio_connector.pt")
    print("Stage 1 complete → data/ckpt/audio_enc.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(epochs=args.epochs)
