import argparse

import torch
from torch.utils.data import DataLoader

from data import ToyAVEDataset, VALUES
from model import AttributeValueExtractor, LLMArena, cleaning_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/synthave_extractor.pt")
    args = parser.parse_args()

    ds = ToyAVEDataset(size=256, seed=123, noise=0.2)
    loader = DataLoader(ds, batch_size=128)
    model = AttributeValueExtractor()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    arena = LLMArena()

    batch = next(iter(loader))
    with torch.no_grad():
        pred = model(batch["features"]).argmax(dim=-1)
        majority, agreement, _ = arena.vote(batch["features"], batch["synthetic_label"])
    metrics = cleaning_metrics(majority, agreement, batch["value_id"], batch["synthetic_label"])
    sample = {
        "text": batch["text"][0],
        "extractor_value": VALUES[int(pred[0])],
        "arena_value": VALUES[int(majority[0])],
        "agreement": round(float(agreement[0]), 3),
    }
    print({"metrics": {k: round(v, 4) for k, v in metrics.items()}, "sample": sample})


if __name__ == "__main__":
    main()
