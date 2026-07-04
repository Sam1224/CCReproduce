import torch
from torch import nn
from torch.utils.data import DataLoader

from data import MirageDataset, collate
from model import MirageLinker


def train(epochs: int = 30) -> MirageLinker:
    dataset = MirageDataset()
    input_size = dataset[0]["chunk_meta"].numel()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    model = MirageLinker(input_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["chunk_meta"], batch["image_meta"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "mirage_linker.pt")
    return model


if __name__ == "__main__":
    train()
