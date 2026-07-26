import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, FINE_LABELS, PAD_ID, XuanwuToyDataset, build_vocab
from model import XuanwuVL2BToy, count_parameters, stage_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy Xuanwu-VL-2B content-governance model.")
    parser.add_argument("--stage", choices=["pre", "mid", "post", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="xuanwu_vl_2b_toy.pt")
    return parser.parse_args()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def run_stage(model: XuanwuVL2BToy, loader: DataLoader, optimizer: torch.optim.Optimizer, stage: str, epochs: int) -> None:
    for epoch in range(epochs):
        model.train()
        total = {"loss": 0.0, "ce": 0.0, "fine": 0.0, "align": 0.0, "consistency": 0.0, "deploy": 0.0}
        total_acc = 0.0
        steps = 0
        for batch in loader:
            outputs = model(
                batch["image"].float(),
                batch["text_tokens"].long(),
                batch["ocr_tokens"].long(),
                batch["adv_ocr_tokens"].long(),
            )
            loss, logs = stage_loss(outputs, batch["label"].long(), batch["fine_labels"].float(), stage=stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            for key in total:
                total[key] += logs[key]
            total_acc += accuracy(outputs["logits"].detach(), batch["label"].long())
            steps += 1
        mean = {key: value / max(steps, 1) for key, value in total.items()}
        print(
            f"stage={stage:4s} epoch={epoch + 1:02d} "
            f"loss={mean['loss']:.4f} ce={mean['ce']:.4f} fine={mean['fine']:.4f} "
            f"cons={mean['consistency']:.4f} deploy={mean['deploy']:.4f} acc={total_acc / max(steps, 1):.4f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = XuanwuToyDataset(num_samples=args.train_samples, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    vocab = build_vocab()
    model = XuanwuVL2BToy(
        vocab_size=len(vocab),
        num_classes=len(CLASS_NAMES),
        fine_dim=len(FINE_LABELS),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        pad_id=PAD_ID,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    stages = ["pre", "mid", "post"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(model, loader, optimizer, stage=stage, epochs=args.epochs)

    output_path = Path(args.output)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "embed_dim": args.embed_dim,
                "num_classes": len(CLASS_NAMES),
                "fine_dim": len(FINE_LABELS),
                "vocab_size": len(vocab),
                "pad_id": PAD_ID,
                "class_names": CLASS_NAMES,
                "fine_labels": FINE_LABELS,
                "seed": args.seed,
                "parameters": count_parameters(model),
            },
        },
        output_path,
    )
    print(f"saved={output_path.resolve()}")
    print(f"trainable_parameters={count_parameters(model)}")


if __name__ == "__main__":
    main()
