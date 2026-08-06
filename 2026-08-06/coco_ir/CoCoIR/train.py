import torch
from torch.utils.data import DataLoader

from dataset import ToyCoCoIRDataset
from model import TIEModel, coco_ir_loss


def train(epochs=5, batch_size=32):
    dataset = ToyCoCoIRDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TIEModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch_index in range(epochs):
        total_loss = 0.0
        for batch in loader:
            logits = model(batch["source_image"], batch["instructions"], batch["candidates"])
            loss = coco_ir_loss(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch_index + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "tie_toy.pt")


if __name__ == "__main__":
    train()
