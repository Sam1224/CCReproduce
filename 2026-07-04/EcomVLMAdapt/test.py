import torch
from torch.utils.data import DataLoader, random_split

from data import EcomVLMDataset, TOKEN_TO_ID, collate
from model import EcomVLMAdapter


def main() -> None:
    dataset = EcomVLMDataset()
    train_size = int(len(dataset) * 0.8)
    _, test_set = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(2))
    loader = DataLoader(test_set, batch_size=64, collate_fn=collate)
    model = EcomVLMAdapter(vocab_size=len(TOKEN_TO_ID), image_feature_size=12)
    try:
        model.load_state_dict(torch.load("ecom_vlm_adapt.pt", map_location="cpu"))
    except FileNotFoundError:
        pass
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image_features"], batch["text_tokens"])
            predicted = torch.sigmoid(outputs["attribute_logits"]) >= 0.5
            correct += (predicted == (batch["target"] >= 0.5)).sum().item()
            total += batch["target"].numel()
    first = next(iter(loader))
    print({"attribute_accuracy": round(correct / total, 4), "json_example": model.extract_json(first["image_features"][:1], first["text_tokens"][:1])[0]})


if __name__ == "__main__":
    main()
