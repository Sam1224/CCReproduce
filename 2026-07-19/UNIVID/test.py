import argparse

import torch
from torch.utils.data import DataLoader

from data import ToyVideoModerationDataset, decode_caption
from model import UNIVIDLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/univid_lite.pt")
    args = parser.parse_args()

    ds = ToyVideoModerationDataset(size=64, seed=99)
    loader = DataLoader(ds, batch_size=16)
    model = UNIVIDLite()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    correct = 0
    total = 0
    examples = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["frame_features"], batch["text_features"], batch["policy_id"])
            pred = (torch.sigmoid(out["violation_logit"]) >= 0.5).long()
            labels = batch["violation_label"].long()
            correct += (pred == labels).sum().item()
            total += labels.numel()
            if not examples:
                examples = decode_caption(out["caption_logits"][0])
    print({"toy_violation_accuracy": round(correct / total, 4), "sample_policy_caption": examples})


if __name__ == "__main__":
    main()
