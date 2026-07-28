import argparse
import os
import random
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import LaRecDataset, collate_fn
from model import LaRecConfig, LaRecModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(split: str, batch_size: int) -> DataLoader:
    dataset = LaRecDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_fn)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model: LaRecModel, loader: DataLoader, optimizer: torch.optim.Optimizer, stage: str, device: torch.device) -> Dict[str, float]:
    model.train()
    totals = {}
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        outputs = model.forward_pretrain(batch) if stage == "pretrain" else model.forward_rl(batch)
        outputs["loss"].backward()
        optimizer.step()
        steps += 1
        for key, value in outputs.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(model: LaRecModel, output_dir: str, config: LaRecConfig) -> str:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "larec.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy LaRec reproduction.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--rl-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LaRecDataset(split="train")
    train_loader = build_loader("train", args.batch_size)
    config = LaRecConfig(num_items=train_dataset.num_items)
    model = LaRecModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("== Stage 1: Latent Pre-training ==")
    for epoch in range(1, args.pretrain_epochs + 1):
        metrics = run_epoch(model, train_loader, optimizer, stage="pretrain", device=device)
        print(
            f"[Pretrain {epoch:02d}] loss={metrics['loss']:.4f} "
            f"rank={metrics['rank_loss']:.4f} step={metrics['step_alignment']:.4f} process={metrics['process_direction']:.4f}"
        )

    print("== Stage 2: Personalized RL-tuning ==")
    for epoch in range(1, args.rl_epochs + 1):
        metrics = run_epoch(model, train_loader, optimizer, stage="rl", device=device)
        print(
            f"[RL {epoch:02d}] loss={metrics['loss']:.4f} "
            f"policy={metrics['policy_loss']:.4f} reward={metrics['reward']:.4f} reg={metrics['regularization']:.4f}"
        )

    checkpoint_path = save_checkpoint(model, args.output_dir, config)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
