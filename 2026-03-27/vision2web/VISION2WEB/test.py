from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import load_tasks
from generator import render_html
from model import TemplateSelector
from verifier import verify_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    tasks = load_tasks(args.test)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = TemplateSelector()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    ok = 0
    n = 0
    for t in tasks:
        x = torch.tensor([t.spec.to_features()], dtype=torch.float32)
        with torch.no_grad():
            tid = int(model(x).argmax(dim=-1).item())

        html = render_html(t.spec, template_id=tid)
        vr = verify_html(t.spec, html)
        ok += 1 if vr.ok else 0
        n += 1

    print(f"tasks={n} pass@1={ok/max(1,n):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
