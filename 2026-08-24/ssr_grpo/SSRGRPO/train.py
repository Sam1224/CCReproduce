import torch
from torch.utils.data import DataLoader

from dataset import RetrievalToyDataset, VOCAB
from model import SSRGRPOModel, ssr_loss


def main() -> None:
    torch.manual_seed(19)
    dataset = RetrievalToyDataset(size=512)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SSRGRPOModel(len(VOCAB))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for epoch in range(12):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = ssr_loss(outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "ssr_grpo_toy.pt")


if __name__ == "__main__":
    main()
