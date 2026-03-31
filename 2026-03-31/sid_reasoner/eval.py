from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticSIDDataset, collate, parse_prediction
from model import CausalTransformerLM


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval", type=int, default=2000)
    args = ap.parse_args()

    ds = SyntheticSIDDataset(n=args.eval, seed=17)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalTransformerLM(vocab=ds.vocab.vocab_size).to(device)
    payload = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()

    eos_id = ds.vocab.stoi["<eos>"]

    hits = 0
    total = 0
    for batch in loader:
        ids = batch["ids"].to(device)
        prompt_len = batch["prompt_len"].to(device)
        target_sid = batch["target_sid"].to(device)

        max_pl = int(prompt_len.max().item())
        prompt = ids[:, :max_pl]
        gen, _ = model.generate(prompt=prompt, max_new_tokens=8, eos_id=eos_id)

        for i in range(gen.shape[0]):
            pred, ok = parse_prediction(gen[i], ds.vocab)
            hits += int(ok and pred == int(target_sid[i]))
            total += 1

    print({"hit@1": hits / max(1, total)})


if __name__ == "__main__":
    main()
