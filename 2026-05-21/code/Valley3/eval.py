"""Evaluation script for Valley3.

Usage:
    python eval.py --ckpt runs/valley3/model.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from valley3 import Valley3Model, OmniECommerceDataset, ECommerceLoss, compute_task_metrics
from train import collate_fn
from valley3.dataset import TASK_TYPES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/valley3/model.pt")
    p.add_argument("--d_enc", type=int, default=64)
    p.add_argument("--d_llm", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Valley3Model(d_enc=args.d_enc, d_llm=args.d_llm)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    ds = OmniECommerceDataset(num_samples=500, seed=99)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = ECommerceLoss()
    total_loss = 0.0
    task_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            task = batch["task"]
            text = batch["text"].to(device)
            image = batch["image"].to(device)
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            task_name = task[0] if isinstance(task, list) else task
            if task_name == "captioning" and labels.dim() == 1:
                labels = labels.unsqueeze(1)
            logits = model(task, text, image, video, audio)
            total_loss += criterion(logits, labels, task).item()
            m = compute_task_metrics(logits, labels, task)
            for k, v in m.items():
                task_metrics[f"{task_name}_{k}"].append(v)

    print(f"Eval loss: {total_loss / len(loader):.4f}")
    print("Per-task metrics:")
    for k, vs in sorted(task_metrics.items()):
        print(f"  {k}: {sum(vs)/len(vs):.4f}")


if __name__ == "__main__":
    main()
