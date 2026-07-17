import argparse
import json
from pathlib import Path

import torch

from symbal import HashTextEncoder, LinearImageEncoder, detect_systematic_misalignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    text_encoder = HashTextEncoder(dim=128)
    image_encoder = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        image_encoder = LinearImageEncoder(checkpoint["input_dim"], dim=128)
        image_encoder.load_state_dict(checkpoint["image_encoder"])
    predictions = detect_systematic_misalignment(records, image_encoder=image_encoder, text_encoder=text_encoder, top_k_text=args.top_k)
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
