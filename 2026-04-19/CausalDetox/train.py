import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToxicNextTokenDataset, ToyVocab, collate
from model import TinyTransformerLM


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ToxicNextTokenDataset(n=6000)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate)

    vocab = ToyVocab()
    model = TinyTransformerLM(vocab_size=vocab.vocab_size, embed_dim=128, num_layers=3, num_heads=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")
        if step >= 300:
            break

    torch.save(model.state_dict(), "tiny_lm.pt")
    print("saved tiny_lm.pt")

    model.eval()
    head_scores = []

    # Measure necessity-style effect: Δ P(bad) on toxic samples when ablating head h.
    with torch.no_grad():
        base_probs = []
        toxic_loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
        for batch in toxic_loader:
            if batch["is_toxic"].sum().item() == 0:
                continue
            input_ids = batch["input_ids"].to(device)
            is_toxic = batch["is_toxic"].to(device)
            logits = model(input_ids)
            p_bad = torch.softmax(logits[:, -1, :], dim=-1)[:, vocab.bad]
            base_probs.append(p_bad[is_toxic == 1])
            if len(base_probs) >= 10:
                break
        base_p = torch.cat(base_probs).mean().item() if base_probs else 0.0

        for h in range(model.num_heads):
            scale = torch.ones(model.num_heads, device=device)
            scale[h] = 0.0

            ablated_probs = []
            for batch in toxic_loader:
                if batch["is_toxic"].sum().item() == 0:
                    continue
                input_ids = batch["input_ids"].to(device)
                is_toxic = batch["is_toxic"].to(device)
                logits = model(input_ids, head_scale=scale)
                p_bad = torch.softmax(logits[:, -1, :], dim=-1)[:, vocab.bad]
                ablated_probs.append(p_bad[is_toxic == 1])
                if len(ablated_probs) >= 10:
                    break
            ablated_p = torch.cat(ablated_probs).mean().item() if ablated_probs else base_p
            head_scores.append({"head": h, "delta_bad_prob": base_p - ablated_p, "base": base_p, "ablated": ablated_p})

    head_scores.sort(key=lambda x: x["delta_bad_prob"], reverse=True)
    selected = [x["head"] for x in head_scores[:2]]

    with open("selected_heads.json", "w", encoding="utf-8") as f:
        json.dump({"selected_heads": selected, "head_scores": head_scores}, f, indent=2)

    print("selected_heads:", selected)


if __name__ == "__main__":
    main()
