from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ProductFamilyToyDataset, TASKS, collate_ptc
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProductFamilyToyDataset(num_samples=96, seed=99)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_ptc)
    model = OneModel().to(device)
    checkpoint = Path("checkpoints/ptc_toy.pt")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    correct = 0
    total = 0
    by_task = {name: [0, 0] for name in TASKS}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(batch)
        pred = out.logits.argmax(dim=-1)
        correct += int((pred == batch["label"]).sum().item())
        total += pred.numel()
        for task_id, ok in zip(batch["task_id"].cpu().tolist(), (pred == batch["label"]).cpu().tolist()):
            by_task[TASKS[task_id]][0] += int(ok)
            by_task[TASKS[task_id]][1] += 1

    print(f"family defect accuracy (toy): {correct / max(total, 1):.3f}")
    for name, (task_correct, task_total) in by_task.items():
        print(f"{name}: {task_correct / max(task_total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
