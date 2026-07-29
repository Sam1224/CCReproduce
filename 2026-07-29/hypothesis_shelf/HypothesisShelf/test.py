import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyShelfDataset, collate_shelves
from model import HypothesisShelfModel


def evaluate(args):
    dataset = ToyShelfDataset(num_users=args.num_users, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_shelves)
    model = HypothesisShelfModel()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    type_hits = []
    shelf_hits = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["profile"], batch["target_type"], batch["catalogue"], batch["catalogue_type"])
            type_hits.append((outputs["type_logits"].argmax(dim=-1) == batch["target_type"]).float())
            hits = (outputs["final_indices"].unsqueeze(-1) == batch["positive_items"].unsqueeze(1)).any(dim=(1, 2)).float()
            shelf_hits.append(hits)
    print(f"type_accuracy: {torch.cat(type_hits).mean().item():.4f}")
    print(f"shelf_hit_rate: {torch.cat(shelf_hits).mean().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num-users", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    evaluate(parser.parse_args())
