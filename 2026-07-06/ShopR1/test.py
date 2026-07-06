import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ACTIONS, ATTRIBUTES, VALUES, ShoppingBehaviorDataset, collate_batch
from model import ShopR1Policy


def evaluate(args):
    dataset = ShoppingBehaviorDataset(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    if checkpoint:
        dataset.vocab.token_to_id = checkpoint["vocab"]
    model = ShopR1Policy(vocab_size=len(dataset.vocab.token_to_id))
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    total = 0
    exact = 0
    with torch.no_grad():
        for batch in loader:
            pred = model.predict(batch["input_ids"])
            match = pred["action"].eq(batch["action_type"]) & pred["attribute"].eq(batch["attribute"]) & pred["value"].eq(batch["value"])
            exact += int(match.sum())
            total += batch["input_ids"].size(0)
    print(f"exact_action_accuracy={exact / max(total, 1):.4f} ({exact}/{total})")

    sample = dataset[0]
    with torch.no_grad():
        pred = model.predict(sample["input_ids"].unsqueeze(0))
    print("sample_prediction", {
        "action": ACTIONS[int(pred["action"][0])],
        "attribute": ATTRIBUTES[int(pred["attribute"][0])],
        "value": VALUES[int(pred["value"][0])],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="toy_shopping.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=3)
    evaluate(parser.parse_args())
