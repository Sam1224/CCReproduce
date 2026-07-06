import argparse

import torch
from torch.utils.data import DataLoader

from dataset import CATEGORIES, RULES, EvadeDataset, collate_batch
from model import EvadeModerator


def evaluate(args):
    dataset = EvadeDataset()
    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    if checkpoint:
        dataset.vocab.token_to_id = checkpoint["vocab"]
    model = EvadeModerator(vocab_size=len(dataset.vocab.token_to_id))
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model.explain(batch["input_ids"], batch["image_features"])
            correct += int(outputs["category"].eq(batch["label"]).sum())
            total += batch["label"].numel()
    print(f"category_accuracy={correct / max(total, 1):.4f} ({correct}/{total})")

    sample = dataset[0]
    with torch.no_grad():
        explanation = model.explain(sample["input_ids"].unsqueeze(0), sample["image_features"].unsqueeze(0))
    active_rules = [RULES[idx] for idx, flag in enumerate(explanation["rules"][0].tolist()) if flag]
    print("sample_prediction", {"category": CATEGORIES[int(explanation["category"][0])], "rules": active_rules})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=3)
    evaluate(parser.parse_args())
