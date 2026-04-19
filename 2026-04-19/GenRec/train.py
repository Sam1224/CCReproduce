import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SyntheticGenRecDataset, collate_batch, GenRecVocab
from model import CausalTransformer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SyntheticGenRecDataset()
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_batch)

    vocab_size = GenRecVocab.sid_base + 97
    model = CausalTransformer(vocab_size=vocab_size, embed_dim=256, num_layers=4, num_heads=8).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    model.train()
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        sep_pos = (input_ids == GenRecVocab.sep_id).int().argmax(dim=1)
        prefix_len = int(sep_pos.max().item()) + 1

        logits = model(input_ids, attention_mask=attention_mask, prefix_len=prefix_len)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")
        if step >= 400:
            break

    torch.save(model.state_dict(), "genrec_sft.pt")
    print("saved genrec_sft.pt")


if __name__ == "__main__":
    main()
