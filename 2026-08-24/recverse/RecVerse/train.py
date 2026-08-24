import torch
from torch.utils.data import DataLoader

from dataset import ACTIONS, ShoppingTrajectoryDataset
from model import RecVerseModel, recverse_loss


def main() -> None:
    torch.manual_seed(23)
    dataset = ShoppingTrajectoryDataset(trajectories=256)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = RecVerseModel(action_count=len(ACTIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = recverse_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "recverse_toy.pt")


if __name__ == "__main__":
    main()
