import torch
from torch.utils.data import DataLoader

from dataset import ATTRIBUTES, VALUES, VERDICTS, VOCAB, ToyCatalogDataset
from model import TRACEModel


def main() -> None:
    torch.manual_seed(11)
    dataset = ToyCatalogDataset(size=96, seed=11)
    loader = DataLoader(dataset, batch_size=32)
    model = TRACEModel(len(VOCAB), len(ATTRIBUTES), len(VALUES), len(VERDICTS))
    state_path = "trace_toy.pt"
    try:
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    except FileNotFoundError:
        print("checkpoint not found; evaluating randomly initialized model")
    model.eval()
    correct_values = 0
    correct_verdicts = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            correct_values += (outputs["value_logits"].argmax(dim=-1) == batch["value"]).sum().item()
            correct_verdicts += (outputs["verdict_logits"].argmax(dim=-1) == batch["verdict"]).sum().item()
            total += batch["value"].numel()
    print({"value_accuracy": correct_values / total, "verdict_accuracy": correct_verdicts / total})


if __name__ == "__main__":
    main()
