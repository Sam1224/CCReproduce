import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import GroceryDataset, TOKEN_TO_ID, CATEGORIES, collate
from model import GrocLM


def train(epochs: int = 8) -> GrocLM:
    dataset = GroceryDataset(size=400)
    train_size = int(len(dataset) * 0.8)
    train_set, _ = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(1))
    loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)
    model = GrocLM(vocab_size=len(TOKEN_TO_ID), category_count=len(CATEGORIES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["history"], batch["query"], batch["rebuy_prior"])
            loss = loss_fn(logits, batch["target"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "groclm_toy.pt")
    return model


if __name__ == "__main__":
    train()
