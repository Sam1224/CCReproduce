import torch
from torch.utils.data import DataLoader
from dataset import DARTRecoveryDataset, get_mock_data
from model import DARTSelfDistillationModel, localized_recovery_loss


def train(epochs: int = 3) -> None:
    teacher_rollouts, student_rollouts = get_mock_data()
    dataset = DARTRecoveryDataset(teacher_rollouts, student_rollouts)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = DARTSelfDistillationModel(vocab_size=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, labels, recovery_mask in loader:
            logits = model(input_ids)
            loss = localized_recovery_loss(logits, labels, recovery_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(len(loader), 1):.4f}")
    torch.save(model.state_dict(), "dart_sd_toy.pt")


if __name__ == "__main__":
    train()
