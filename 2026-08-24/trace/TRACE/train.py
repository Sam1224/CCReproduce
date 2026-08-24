import torch
from torch.utils.data import DataLoader

from dataset import ATTRIBUTES, VALUES, VERDICTS, VOCAB, ToyCatalogDataset
from model import TRACEModel, trace_loss


def main() -> None:
    torch.manual_seed(7)
    dataset = ToyCatalogDataset(size=384)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TRACEModel(len(VOCAB), len(ATTRIBUTES), len(VALUES), len(VERDICTS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(8):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = trace_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "trace_toy.pt")


if __name__ == "__main__":
    main()
