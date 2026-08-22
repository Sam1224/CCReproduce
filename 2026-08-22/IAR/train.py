import torch
from torch import nn

from dataset import build_iar_examples, encode
from model import TinyIARModel, merge_recover


def train_stage(model, rows, vocab, stage_name, epochs=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    stage_rows = [row for row in rows if row[2].startswith(stage_name)]
    for epoch in range(epochs):
        total = 0.0
        for prompt, target, _ in stage_rows:
            x = torch.tensor([encode(prompt, vocab)])
            first_target_token = encode(target, vocab, max_len=1)[0]
            y = torch.tensor([first_target_token])
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if stage_rows and (epoch == epochs - 1 or epoch % 10 == 0):
            print(f"{stage_name} epoch={epoch:02d} loss={total / len(stage_rows):.4f}")


def main():
    torch.manual_seed(11)
    vocab, rows = build_iar_examples()
    base_model = TinyIARModel(len(vocab))
    domain_model = TinyIARModel(len(vocab))
    domain_model.load_state_dict(base_model.state_dict())
    train_stage(domain_model, rows, vocab, "inject", epochs=35)
    train_stage(domain_model, rows, vocab, "align", epochs=35)
    recovered_state = merge_recover(domain_model, base_model, alpha=0.7)
    recovered_model = TinyIARModel(len(vocab))
    recovered_model.load_state_dict(recovered_state)
    torch.save({"model": recovered_model.state_dict(), "vocab": vocab}, "iar_toy.pt")
    print("saved iar_toy.pt")


if __name__ == "__main__":
    main()
