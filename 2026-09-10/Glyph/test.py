import argparse
from pathlib import Path

import torch

from data import CharTokenizer
from model import ContrastiveMetadataTagger, GlyphPipeline


def load_model(checkpoint_path: Path, tokenizer: CharTokenizer, dim: int):
    model = ContrastiveMetadataTagger(vocab_size=tokenizer.vocab_size, dim=dim)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def main(args):
    tokenizer = CharTokenizer()
    model = load_model(Path(args.checkpoint), tokenizer, args.dim)
    pipeline = GlyphPipeline(model, tokenizer)
    samples = [
        ("commerce.risk.creators.creator_penalty_score_12", "creator governance risk level", "commerce"),
        ("ads.profile.users.email_hash_5", "email address for account communication", "ads"),
        ("commerce.items.catalog.brand_value_8", "product catalogue attribute", "commerce"),
    ]
    for metadata_key, description, line_of_business in samples:
        predictions = pipeline.predict(metadata_key, description, line_of_business)
        pretty = ", ".join(f"{tag}:{score:.3f}" for tag, score, _ in predictions)
        print(f"{metadata_key} -> {pretty}")
    print("Glyph pipeline smoke test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test Glyph-style multi-strategy tagging")
    parser.add_argument("--checkpoint", default="artifacts/glyph_toy.pt")
    parser.add_argument("--dim", type=int, default=128)
    main(parser.parse_args())
