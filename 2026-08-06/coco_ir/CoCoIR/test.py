import torch
from torch.utils.data import DataLoader

from dataset import ToyCoCoIRDataset
from model import TIEModel


def main():
    dataset = ToyCoCoIRDataset(num_dialogues=64, seed=17)
    loader = DataLoader(dataset, batch_size=32)
    model = TIEModel()
    model.load_state_dict(torch.load("tie_toy.pt", map_location="cpu"))
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["source_image"], batch["instructions"], batch["candidates"])
            prediction = logits.argmax(dim=-1)
            correct += (prediction == batch["labels"]).sum().item()
            total += batch["labels"].numel()
    print({"turn_level_recall@1": correct / total})


if __name__ == "__main__":
    main()
