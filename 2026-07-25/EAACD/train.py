import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToyQADataset, Vocabulary, build_toy_samples
from model import EAACDConfig, ToyMoELanguageModel


def train(args):
    samples = build_toy_samples()
    vocab = Vocabulary.build(samples)
    dataset = ToyQADataset(samples, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = EAACDConfig(vocab_size=len(vocab.token_to_id), hidden_size=args.hidden_size, num_experts=args.num_experts)
    model = ToyMoELanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            outputs = model(batch["input_ids"])
            first_answer_token = batch["target_ids"][:, 0]
            lm_loss = F.cross_entropy(outputs["logits"], first_answer_token)
            factual_loss = F.binary_cross_entropy_with_logits(outputs["factual_logit"], batch["is_factual"])
            router_balance = outputs["router_probs"].mean(dim=(0, 1, 2)).var()
            loss = lm_loss + 0.4 * factual_loss + 0.02 * router_balance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(json.dumps({"epoch": epoch + 1, "loss": round(total_loss / len(loader), 4)}))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__, "vocab": vocab.token_to_id}, output_dir / "checkpoint.pt")
    print(f"saved {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output-dir", type=str, default="runs/eaacd_toy")
    train(parser.parse_args())
