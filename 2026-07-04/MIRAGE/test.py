import torch
from torch.utils.data import DataLoader

from data import MirageDataset, collate
from model import MirageLinker, evaluate_alignment, generate_answer


def main() -> None:
    dataset = MirageDataset()
    input_size = dataset[0]["chunk_meta"].numel()
    model = MirageLinker(input_size)
    try:
        model.load_state_dict(torch.load("mirage_linker.pt", map_location="cpu"))
    except FileNotFoundError:
        pass
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probability = torch.sigmoid(model(batch["chunk_meta"], batch["image_meta"]))
            correct += ((probability >= 0.5) == (batch["label"] >= 0.5)).sum().item()
            total += batch["label"].numel()
    sample = generate_answer("open sim tray", {"product": "phone", "aspect": "sim tray", "context": "cannot receive calls"})
    metrics = evaluate_alignment(
        {"product": "phone", "aspect": "sim tray", "context": "cannot receive calls"},
        {"product": "phone", "aspect": "sim tray", "context": "cannot receive calls"},
    )
    print({"link_accuracy": round(correct / total, 4), "answer": sample, "m_eval": metrics})


if __name__ == "__main__":
    main()
