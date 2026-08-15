import torch

from data import ToyEcommerceSequenceDataset
from tm20k import TM20KEncoder, TM20KRanker, TM20KStudent, TokenMerger, tm20k_distillation_loss


def test_forward_and_distillation_step():
    dataset = ToyEcommerceSequenceDataset(num_samples=8, seq_len=64)
    batch = {key: torch.stack([dataset[i][key] for i in range(4)]) for key in ["tokens", "categories", "positions", "label"]}
    teacher = TM20KRanker(TM20KEncoder(max_seq_len=64, hidden_size=32, num_layers=1, num_heads=4))
    student = TM20KStudent(
        TM20KRanker(TM20KEncoder(max_seq_len=64, hidden_size=32, num_layers=1, num_heads=4)),
        TokenMerger(strategy="recent_keep", merge_ratio=0.25),
    )
    teacher_output = teacher(batch["tokens"], batch["categories"], batch["positions"])
    student_output = student(batch["tokens"], batch["categories"], batch["positions"])
    assert teacher_output["logits"].shape == (4, 2)
    assert student_output["sequence"].shape[1] == 16
    loss = tm20k_distillation_loss(student_output, teacher_output, batch["label"])
    loss.backward()
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_forward_and_distillation_step()
    print("TM20K smoke test passed")
