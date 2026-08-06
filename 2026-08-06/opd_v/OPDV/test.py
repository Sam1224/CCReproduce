import torch
from torch.utils.data import DataLoader

from dataset import ToyVisualReasoningDataset
from model import TinyMLLM, modality_balance_ratio


def main():
    dataset = ToyVisualReasoningDataset(num_samples=64, seed=31)
    loader = DataLoader(dataset, batch_size=32)
    model = TinyMLLM()
    model.load_state_dict(torch.load("opdv_toy.pt", map_location="cpu"))
    correct = 0
    total = 0
    ratios = []
    with torch.no_grad():
        for batch in loader:
            logits, attention = model(batch["original_image"], batch["text"])
            prediction = logits.argmax(dim=-1)
            correct += (prediction == batch["labels"]).sum().item()
            total += batch["labels"].numel()
            ratios.append(modality_balance_ratio(attention).mean().item())
    print({"token_accuracy": correct / total, "visual_text_attention_ratio": sum(ratios) / len(ratios)})


if __name__ == "__main__":
    main()
