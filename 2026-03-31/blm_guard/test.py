from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from dataset import make_dataset, split
from model import BLMGuardModel, aks_bin_top, select_regions


def f1_binary(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))


def main() -> None:
    data = make_dataset(n=2000, seed=7)
    _, te = split(data, 0.0)

    model = BLMGuardModel()
    ckpt = Path("checkpoints/blm_guard.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    model.eval()

    y_r = []
    p_r = []

    wide = []
    strict = []

    for ex in te:
        scores = ex.frames.norm(dim=-1)
        idx = aks_bin_top(scores, m=3)
        region = select_regions(ex.frames[idx]).mean(dim=0)

        s_logits, t_logits, r_logit = model(region, ex.asr)
        s_pred = int(torch.argmax(s_logits).item())
        t_pred = int(torch.argmax(t_logits).item())
        r_pred = int((torch.sigmoid(r_logit) > 0.5).item())

        y_r.append(ex.label_risky)
        p_r.append(r_pred)

        strict.append(int(s_pred == ex.label_scene and t_pred == ex.label_type))
        wide.append(int(s_pred == ex.label_scene or t_pred == ex.label_type))

    y_r = np.array(y_r, dtype=np.int64)
    p_r = np.array(p_r, dtype=np.int64)

    print(f"Strict accuracy (scene&type): {float(np.mean(strict)):.3f}")
    print(f"Wide accuracy (scene|type): {float(np.mean(wide)):.3f}")
    print(f"Risky F1: {f1_binary(y_r, p_r):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
