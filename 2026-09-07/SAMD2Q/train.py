from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToySAMD2QDataset
from model import SAMD2Q, business_reward_loss, sequence_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/samd2q.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToySAMD2QDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SAMD2Q(vocab_size=dataset.vocab_size, image_dim=dataset.image_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch, use_masked_title=False)
            sft = sequence_loss(logits, batch["target_query"])
            cf_logits = model(batch, use_masked_title=True)
            counterfactual = sequence_loss(cf_logits, batch["target_query"])
            reward_loss, _ = business_reward_loss(model, batch)
            loss = sft + 0.7 * counterfactual + 0.1 * reward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch} loss={total / len(loader):.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": dataset.vocab_size, "image_dim": dataset.image_dim}, output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
