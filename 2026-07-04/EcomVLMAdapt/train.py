import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import EcomVLMDataset, TOKEN_TO_ID, collate
from model import EcomVLMAdapter


def train(epochs: int = 8) -> EcomVLMAdapter:
    dataset = EcomVLMDataset()
    train_size = int(len(dataset) * 0.8)
    train_set, _ = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(2))
    loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)
    model = EcomVLMAdapter(vocab_size=len(TOKEN_TO_ID), image_feature_size=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["image_features"], batch["text_tokens"])
            loss = loss_fn(outputs["attribute_logits"], batch["target"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "ecom_vlm_adapt.pt")
    return model


if __name__ == "__main__":
    train()
