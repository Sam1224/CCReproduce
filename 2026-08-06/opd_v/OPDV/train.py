import copy
import torch
from torch.utils.data import DataLoader

from dataset import ToyVisualReasoningDataset
from model import TinyMLLM, opd_v_loss


def update_ema(teacher, student, momentum=0.98):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def train(epochs=4, batch_size=32):
    dataset = ToyVisualReasoningDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    student = TinyMLLM()
    teacher = copy.deepcopy(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    for epoch_index in range(epochs):
        total_loss = 0.0
        for batch in loader:
            student_logits, student_attention = student(batch["original_image"], batch["text"])
            with torch.no_grad():
                positive_logits, positive_attention = teacher(batch["zoom_image"], batch["text"])
                negative_logits, negative_attention = teacher(batch["mask_image"], batch["text"])
            loss = opd_v_loss(student_logits, positive_logits, negative_logits, batch["labels"], positive_attention, negative_attention)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(teacher, student)
            total_loss += loss.item()
        print(f"epoch={epoch_index + 1} loss={total_loss / len(loader):.4f}")
    torch.save(student.state_dict(), "opdv_toy.pt")


if __name__ == "__main__":
    train()
