import argparse
from pathlib import Path

import torch

from dataset import ToyQADataset, Vocabulary, build_toy_samples
from model import EAACDConfig, EAACDDecoder, ToyMoELanguageModel


def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    token_to_id = checkpoint["vocab"]
    vocab = Vocabulary(token_to_id=token_to_id, id_to_token={index: token for token, index in token_to_id.items()})
    config = EAACDConfig(**checkpoint["config"])
    model = ToyMoELanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab, config


def evaluate(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}; run train.py first")
    model, vocab, config = load_model(str(checkpoint))
    decoder = EAACDDecoder(model, config)
    samples = build_toy_samples()
    dataset = ToyQADataset(samples, vocab)
    exact_first_token = 0
    factual_scores = []
    for item, raw in zip(dataset, samples):
        input_ids = item["input_ids"].unsqueeze(0)
        decoded = decoder.contrast_logits(input_ids)
        prediction = int(decoded["logits"].argmax(dim=-1).item())
        target = int(item["target_ids"][0].item())
        exact_first_token += int(prediction == target)
        factual_scores.append(float(decoded["factual_prob"].item()))
        print({
            "question": raw["question"],
            "target": raw["answer"],
            "predicted_first_token": vocab.id_to_token[prediction],
            "factual_prob": round(float(decoded["factual_prob"].item()), 4),
            "negative_strength": round(float(decoded["negative_strength"].item()), 4),
        })
    print({"first_token_accuracy": round(exact_first_token / len(dataset), 4), "mean_factual_prob": round(sum(factual_scores) / len(factual_scores), 4)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/eaacd_toy/checkpoint.pt")
    evaluate(parser.parse_args())
