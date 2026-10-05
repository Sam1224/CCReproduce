import argparse

import torch
from torch.utils.data import DataLoader

from dataset import PREFERRED, SOURCES, SourcePreferenceDataset, make_item
from model import SourceAwareRanker


def preference_gap(model, device):
    gaps = []
    for preferred in PREFERRED:
        for dispreferred in set(SOURCES) - PREFERRED:
            left_features, left_source, _ = make_item(4, 3, preferred, "shopping")
            right_features, right_source, _ = make_item(4, 3, dispreferred, "shopping")
            batch = {
                "left_features": left_features.unsqueeze(0).to(device),
                "right_features": right_features.unsqueeze(0).to(device),
                "left_source": torch.tensor([left_source], device=device),
                "right_source": torch.tensor([right_source], device=device),
            }
            with torch.no_grad():
                gaps.append(torch.sigmoid(model(batch)["pair_logit"]).item() - 0.5)
    return sum(gaps) / len(gaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/demo/debiased.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)
    model = SourceAwareRanker(num_sources=len(payload.get("sources", SOURCES))).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    data = SourcePreferenceDataset(size=512, seed=99, bias_strength=0.0)
    loader = DataLoader(data, batch_size=128)
    correct = total = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            pred = (model(batch)["pair_logit"] >= 0).float()
        correct += int((pred == batch["label"]).sum())
        total += int(batch["label"].numel())
    print({"utility_accuracy": correct / total, "matched_source_preference_gap": preference_gap(model, device)})


if __name__ == "__main__":
    main()
