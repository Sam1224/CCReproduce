import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ToyFashionDetailDataset
from model import DetailAnywhere


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = ToyFashionDetailDataset(length=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = DetailAnywhere().to(device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    l1_scores = []
    color_scores = []
    with torch.no_grad():
        for batch in loader:
            reference = batch["reference"].to(device)
            focus_mask = batch["focus_mask"].to(device)
            target_detail = batch["target_detail"].to(device)
            category_id = batch["category_id"].to(device)
            detail = model(reference, focus_mask, category_id)["detail"]
            l1_scores.append(F.l1_loss(detail, target_detail).item())
            color_scores.append(F.mse_loss(detail.mean(dim=(2, 3)), target_detail.mean(dim=(2, 3))).item())
    print({"toy_l1": sum(l1_scores) / len(l1_scores), "toy_color_mse": sum(color_scores) / len(color_scores)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/detailanywhere_toy.pt")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())
