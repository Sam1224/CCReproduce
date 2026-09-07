from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import PAD, ToySAMD2QDataset
from model import SAMD2Q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/samd2q.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToySAMD2QDataset(num_samples=256)
    model = SAMD2Q(vocab_size=dataset.vocab_size, image_dim=dataset.image_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    token_hits, token_total, exact = 0, 0, 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size):
            batch = {key: value.to(device) for key, value in batch.items()}
            generated = model.generate(batch["masked_title"], batch["image"], max_len=batch["target_query"].size(1))
            target = batch["target_query"][:, 1:]
            mask = target != PAD
            token_hits += int(((generated == target) & mask).sum().item())
            token_total += int(mask.sum().item())
            exact += int((((generated == target) | ~mask).all(dim=1)).sum().item())
    print({"token_accuracy": round(token_hits / max(token_total, 1), 4), "exact_match": round(exact / len(dataset), 4)})


if __name__ == "__main__":
    main()
