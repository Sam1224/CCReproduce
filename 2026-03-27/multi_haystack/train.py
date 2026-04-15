import torch
from torch.utils.data import DataLoader

from dataset import ToyMultiHaystackDataset, collate_fn
from model import MultiHaystackRetriever, info_nce_loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ToyMultiHaystackDataset(num_queries=256, num_docs=512)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = MultiHaystackRetriever().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(dl):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            loss = info_nce_loss(out["scores"], batch["label"])

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    train()
