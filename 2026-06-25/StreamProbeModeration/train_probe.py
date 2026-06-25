from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToyModerationConfig, ToyModerationDataset
from model import HiddenStateProbe, ToyCausalTransformer, ToyLMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--probe-layer", type=int, default=3, help="which hidden state to probe (0=emb, 1..n_layers)")
    p.add_argument("--train-n", type=int, default=5000)
    p.add_argument("--val-n", type=int, default=1000)
    return p.parse_args()


@torch.no_grad()
def eval_f1(model, probe, loader, layer: int):
    model.eval()
    probe.eval()
    tp = fp = fn = 0
    for batch in loader:
        x = batch["x"]
        y = batch["y_stream"]
        _, hiddens = model(x, return_hiddens=True)
        h = hiddens[layer]
        p = probe(h)
        pred = (p > 0.5).to(torch.int64)
        gold = y.to(torch.int64)
        tp += int(((pred == 1) & (gold == 1)).sum().item())
        fp += int(((pred == 1) & (gold == 0)).sum().item())
        fn += int(((pred == 0) & (gold == 1)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = ToyModerationConfig()
    lm_cfg = ToyLMConfig(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len)

    train_ds = ToyModerationDataset(args.train_n, cfg, seed=args.seed)
    val_ds = ToyModerationDataset(args.val_n, cfg, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ToyCausalTransformer(lm_cfg, trigger_tokens=cfg.trigger_tokens)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if not (0 <= args.probe_layer <= lm_cfg.n_layers):
        raise ValueError(f"probe_layer must be in [0,{lm_cfg.n_layers}]")

    probe = HiddenStateProbe(lm_cfg.d_model)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    best_f1 = -1.0
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "probe.pt"

    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"]
            y = batch["y_stream"]

            with torch.no_grad():
                _, hiddens = model(x, return_hiddens=True)
                h = hiddens[args.probe_layer]

            p_hat = probe(h)

            # The streaming label is highly imbalanced (mostly 0s before the first trigger).
            # Use a simple per-batch positive reweighting to make training stable.
            pos = float(y.mean().item())
            pos_weight = (1.0 - pos) / max(1e-6, pos)
            w = y * pos_weight + (1.0 - y)
            loss = F.binary_cross_entropy(p_hat, y, weight=w)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        prec, rec, f1 = eval_f1(model, probe, val_loader, args.probe_layer)
        print(
            f"epoch={epoch} loss={total_loss/len(train_loader):.4f} "
            f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "seed": args.seed,
                    "probe_state_dict": probe.state_dict(),
                    "probe_layer": args.probe_layer,
                    "cfg": {
                        "vocab_size": cfg.vocab_size,
                        "seq_len": cfg.seq_len,
                        "trigger_tokens": list(cfg.trigger_tokens),
                    },
                    "lm_cfg": {
                        "d_model": lm_cfg.d_model,
                        "n_layers": lm_cfg.n_layers,
                    },
                },
                ckpt_path,
            )

    print(f"saved: {ckpt_path} (best_f1={best_f1:.4f})")


if __name__ == "__main__":
    main()
