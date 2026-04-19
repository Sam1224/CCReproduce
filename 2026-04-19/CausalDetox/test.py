import json

import torch

from dataset import ToxicNextTokenDataset, ToyVocab, collate
from model import TinyTransformerLM
from torch.utils.data import DataLoader


def mean_bad_prob(model: TinyTransformerLM, loader: DataLoader, head_scale=None) -> float:
    vocab = ToyVocab()
    probs = []
    with torch.no_grad():
        for batch in loader:
            if batch["is_toxic"].sum().item() == 0:
                continue
            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            is_toxic = batch["is_toxic"].to(next(model.parameters()).device)
            logits = model(input_ids, head_scale=head_scale)
            p_bad = torch.softmax(logits[:, -1, :], dim=-1)[:, vocab.bad]
            probs.append(p_bad[is_toxic == 1])
            if len(probs) >= 20:
                break
    return torch.cat(probs).mean().item() if probs else 0.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = ToyVocab()

    model = TinyTransformerLM(vocab_size=vocab.vocab_size, embed_dim=128, num_layers=3, num_heads=4).to(device)
    model.load_state_dict(torch.load("tiny_lm.pt", map_location=device))
    model.eval()

    with open("selected_heads.json", "r", encoding="utf-8") as f:
        selected = json.load(f)["selected_heads"]

    ds = ToxicNextTokenDataset(n=2000)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    base = mean_bad_prob(model, loader)

    scale = torch.ones(model.num_heads, device=device)
    for h in selected:
        scale[h] = 0.3

    detox = mean_bad_prob(model, loader, head_scale=scale)

    print(f"mean_p(bad) toxic prompts: base={base:.3f} detox={detox:.3f} (selected_heads={selected})")


if __name__ == "__main__":
    main()
