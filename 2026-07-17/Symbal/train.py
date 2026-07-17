import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from symbal import HashTextEncoder, LinearImageEncoder, split_facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--checkpoint", default="symbal_projection.pt")
    args = parser.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    image_features = torch.tensor([r["image_features"] for r in records], dtype=torch.float32)
    positive_texts = [split_facts(r["caption"])[0] for r in records]
    text_encoder = HashTextEncoder(dim=128)
    image_encoder = LinearImageEncoder(image_features.shape[1], dim=128)
    optimizer = torch.optim.AdamW(image_encoder.parameters(), lr=1e-2, weight_decay=1e-4)

    text_emb = text_encoder(positive_texts).detach()
    for epoch in range(args.epochs):
        image_emb = image_encoder(image_features)
        logits = image_emb @ text_emb.T / 0.07
        labels = torch.arange(len(records))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch} loss={loss.item():.4f}")

    torch.save({"image_encoder": image_encoder.state_dict(), "input_dim": image_features.shape[1]}, args.checkpoint)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
