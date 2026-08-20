import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OBJECT_TOKENS, ToyCommerceCaptionDataset, VOCAB, collate_batch
from model import build_toy_model
from reweigh import ReWEIGHConfig, ReWEIGHState, greedy_decode_with_reweigh


def greedy_decode(model, image, prompt_ids, max_new_tokens):
    generated = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(image, generated)["logits"][:, -1, :]
            generated = torch.cat([generated, logits.argmax(dim=-1, keepdim=True)], dim=-1)
    return generated


def hallucination_rate(generated, object_ids):
    total = 0
    hallucinated = 0
    for row, objects in zip(generated.tolist(), object_ids.tolist()):
        present = set(objects)
        for token in row:
            if token in OBJECT_TOKENS:
                total += 1
                hallucinated += int(token not in present)
    return hallucinated / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate toy ReWEIGH hallucination mitigation.")
    parser.add_argument("--state", type=str, default="artifacts/reweigh_state.pt")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=2.5)
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.state), map_location="cpu")
    state = ReWEIGHState(reference=checkpoint["reference"], stable_mask=checkpoint["stable_mask"])
    dataset = ToyCommerceCaptionDataset(size=args.samples, seed=31)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)
    model = build_toy_model(vocab_size=len(VOCAB))

    base_rates = []
    reweigh_rates = []
    prompt = torch.tensor([[VOCAB.index("<bos>"), VOCAB.index("a")]], dtype=torch.long)
    for batch in loader:
        prompts = prompt.expand(batch["image"].size(0), -1).clone()
        base = greedy_decode(model, batch["image"], prompts, args.max_new_tokens)
        adjusted = greedy_decode_with_reweigh(model, batch["image"], prompts, state, args.max_new_tokens, ReWEIGHConfig(alpha=args.alpha, topk_reference=len(VOCAB)))
        base_rates.append(hallucination_rate(base, batch["object_ids"]))
        reweigh_rates.append(hallucination_rate(adjusted, batch["object_ids"]))

    base_mean = sum(base_rates) / len(base_rates)
    reweigh_mean = sum(reweigh_rates) / len(reweigh_rates)
    print(f"baseline_hallucination_rate={base_mean:.4f}")
    print(f"reweigh_hallucination_rate={reweigh_mean:.4f}")
    print(f"relative_reduction={(base_mean - reweigh_mean) / max(base_mean, 1e-8):.2%}")


if __name__ == "__main__":
    main()
