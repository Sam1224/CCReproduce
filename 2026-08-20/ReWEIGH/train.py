import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyCommerceCaptionDataset, VOCAB, collate_batch
from model import build_toy_model
from reweigh import ReWEIGHCalibrator, ReWEIGHConfig


def main():
    parser = argparse.ArgumentParser(description="Calibrate ReWEIGH token references on unlabeled toy images.")
    parser.add_argument("--output", type=str, default="artifacts/reweigh_state.pt")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.5)
    args = parser.parse_args()

    dataset = ToyCommerceCaptionDataset(size=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = build_toy_model(vocab_size=len(VOCAB))
    calibrator = ReWEIGHCalibrator(ReWEIGHConfig(alpha=args.alpha, topk_reference=len(VOCAB)))

    evidence_batches = []
    with torch.no_grad():
        for batch in loader:
            visual_hidden = model.encode_image(batch["image"])
            evidence_batches.append(model.visual_readout(visual_hidden))
    state = calibrator.fit(evidence_batches)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"reference": state.reference, "stable_mask": state.stable_mask, "vocab": VOCAB, "alpha": args.alpha}, output)
    print(f"saved {output}")
    print(f"registered_tokens={int(state.stable_mask.sum())}/{len(VOCAB)}")


if __name__ == "__main__":
    main()
