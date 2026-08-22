import torch
from torch import nn

from dataset import step_vocab, toy_policy_cases
from model import ProactiveVerifier, bitmap_for_steps


def build_training_rows():
    cases = toy_policy_cases()
    vocab = step_vocab(cases)
    rows = []
    for case in cases:
        completed = [turn.action for turn in case.transcript if turn.action]
        target = vocab.get(case.expected_next_step, vocab[case.mutation_action])
        rows.append((bitmap_for_steps(completed, vocab), target))
    return vocab, rows


def train(epochs: int = 80):
    torch.manual_seed(7)
    vocab, rows = build_training_rows()
    model = ProactiveVerifier(num_steps=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    xs = torch.stack([row[0] for row in rows])
    ys = torch.tensor([row[1] for row in rows], dtype=torch.long)
    for epoch in range(epochs):
        logits = model(xs)
        loss = criterion(logits, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict(), "vocab": vocab}, "policyguide.pt")
    return model, vocab


if __name__ == "__main__":
    train()
