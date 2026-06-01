#!/usr/bin/env python
"""Stage 2: Cross-Modal Instruction Following (Valley3 toy)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from src.omni_model import OmniEcommerceModel

def generate_toy_ecom_data(n=100, vocab_size=256, d_model=128):
    """Toy e-commerce multimodal instruction-following data."""
    data = []
    for i in range(n):
        label = i % 4   # e.g., product category
        text_embed = torch.randn(8, d_model) + label * 0.3  # [T, d]
        img_feat = torch.randn(16, 64) + label * 0.2        # [patches, img_dim]
        mel = torch.randn(64, 80) + label * 0.1             # [T, mel_bins]
        input_ids = torch.randint(0, vocab_size - 1, (5,))  # target tokens
        target_ids = torch.cat([input_ids[1:],
                                 torch.tensor([label % vocab_size])])
        data.append((text_embed, img_feat, mel, input_ids, target_ids))
    return data

def train(epochs=3, device="cpu"):
    print("Stage 2: Cross-Modal Instruction Following")
    model = OmniEcommerceModel(vocab_size=256, d_model=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    data = generate_toy_ecom_data()

    for epoch in range(epochs):
        total_loss = 0.0
        for text_e, img_f, mel, inp, tgt in data:
            text_e = text_e.unsqueeze(0).to(device)
            img_f = img_f.unsqueeze(0).to(device)
            mel = mel.unsqueeze(0).to(device)
            inp = inp.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)

            logits = model(inp, text_e, img_f, mel)     # [1, T, vocab]
            loss = F.cross_entropy(logits.view(-1, 256), tgt.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: avg_loss={total_loss/len(data):.4f}")

    os.makedirs("data/ckpt", exist_ok=True)
    torch.save(model.state_dict(), "data/ckpt/valley3_toy.pt")
    print("Stage 2 complete → data/ckpt/valley3_toy.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(epochs=args.epochs)
