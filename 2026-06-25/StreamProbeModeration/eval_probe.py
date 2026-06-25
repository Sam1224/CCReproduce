from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import ToyModerationConfig, ToyModerationDataset
from model import HiddenStateProbe, ToyCausalTransformer, ToyLMConfig


@torch.no_grad()
def early_detection_gap(pred_stream: torch.Tensor, gold_stream: torch.Tensor) -> float:
    """Average token gap between first gold unsafe token and first predicted unsafe token.

    If the sample is always safe, we ignore it.
    If the prediction never fires for an unsafe sample, we count it as seq_len.

    Lower is better.
    """
    b, t = pred_stream.shape
    gaps = []
    for i in range(b):
        g = gold_stream[i]
        if g.max().item() < 0.5:
            continue
        first_gold = int(torch.argmax(g).item())
        p = pred_stream[i]
        if p.max().item() < 0.5:
            gaps.append(t)
            continue
        first_pred = int(torch.argmax(p).item())
        gaps.append(max(0, first_pred - first_gold))
    if not gaps:
        return 0.0
    return float(sum(gaps) / len(gaps))


@torch.no_grad()
def f1(pred: torch.Tensor, gold: torch.Tensor) -> tuple[float, float, float]:
    pred = pred.to(torch.int64)
    gold = gold.to(torch.int64)
    tp = int(((pred == 1) & (gold == 1)).sum().item())
    fp = int(((pred == 1) & (gold == 0)).sum().item())
    fn = int(((pred == 0) & (gold == 1)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1v = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1v


def main() -> None:
    here = Path(__file__).resolve().parent
    ckpt_path = here / "checkpoints" / "probe.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}. Run train_probe.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    torch.manual_seed(int(ckpt.get("seed", 0)))
    probe_layer = int(ckpt["probe_layer"])

    cfg = ToyModerationConfig(
        vocab_size=int(ckpt["cfg"]["vocab_size"]),
        seq_len=int(ckpt["cfg"]["seq_len"]),
        trigger_tokens=tuple(int(x) for x in ckpt["cfg"]["trigger_tokens"]),
    )
    lm_cfg = ToyLMConfig(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        d_model=int(ckpt["lm_cfg"]["d_model"]),
        n_layers=int(ckpt["lm_cfg"]["n_layers"]),
    )

    model = ToyCausalTransformer(lm_cfg, trigger_tokens=cfg.trigger_tokens)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    probe = HiddenStateProbe(lm_cfg.d_model)
    probe.load_state_dict(ckpt["probe_state_dict"])
    probe.eval()

    test_ds = ToyModerationDataset(1500, cfg, seed=123)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    all_pred = []
    all_gold = []
    for batch in loader:
        x = batch["x"]
        y = batch["y_stream"]
        _, hiddens = model(x, return_hiddens=True)
        h = hiddens[probe_layer]
        p = probe(h)
        all_pred.append((p > 0.5).to(torch.int64))
        all_gold.append(y.to(torch.int64))

    pred = torch.cat(all_pred, dim=0)
    gold = torch.cat(all_gold, dim=0)

    precision, recall, f1v = f1(pred, gold)
    gap = early_detection_gap(pred.to(torch.float32), gold.to(torch.float32))

    print(f"probe_layer={probe_layer}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1v:.4f}")
    print(f"avg_early_detection_gap_tokens={gap:.2f} (lower is better)")


if __name__ == "__main__":
    main()
