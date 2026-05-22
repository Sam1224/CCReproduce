import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyAIGenDataset, compute_radial_spectrum
from model import FrequencyTeacher, SpatialDetector, distill_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--bins", type=int, default=64)
    args = parser.parse_args()

    ds = ToyAIGenDataset(n=1200, image_size=64, seed=0)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    teacher = FrequencyTeacher(bins=args.bins, dim=128)
    spatial = SpatialDetector(dim=128)

    opt = torch.optim.AdamW(list(teacher.parameters()) + list(spatial.parameters()), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    teacher.train()
    spatial.train()

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for x, y in dl:
            y = y.float()
            spec = torch.stack([compute_radial_spectrum(img, bins=args.bins) for img in x])

            t_z, t_logit = teacher(spec)
            s_z, s_logit = spatial(x)

            loss_cls = bce(s_logit, y) + bce(t_logit, y)
            loss_distill = distill_loss(s_z, t_z.detach())
            loss = loss_cls + 0.5 * loss_distill

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"epoch {epoch}: loss={total/len(dl):.4f}")

    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": spatial.state_dict()}, out_dir / "spatial.pt")
    print("saved spatial -> checkpoints/spatial.pt")


if __name__ == "__main__":
    main()
