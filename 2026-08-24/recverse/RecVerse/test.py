import torch
from torch.utils.data import DataLoader

from dataset import ACTIONS, ShoppingTrajectoryDataset
from model import RecVerseModel


def main() -> None:
    dataset = ShoppingTrajectoryDataset(trajectories=64, seed=29)
    loader = DataLoader(dataset, batch_size=32)
    model = RecVerseModel(action_count=len(ACTIONS))
    try:
        model.load_state_dict(torch.load("recverse_toy.pt", map_location="cpu"))
    except FileNotFoundError:
        print("checkpoint not found; evaluating randomly initialized model")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            predictions = outputs["action_logits"].argmax(dim=-1)
            correct += predictions.eq(batch["actions"]).sum().item()
            total += batch["actions"].numel()
    print({"action_accuracy": correct / total})
    sample = dataset[0]
    with torch.no_grad():
        outputs = model({key: value.unsqueeze(0) for key, value in sample.items()})
        decoded = [ACTIONS[index] for index in outputs["action_logits"].argmax(dim=-1).squeeze(0).tolist()]
    print({"sample_policy": decoded})


if __name__ == "__main__":
    main()
