import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    metrics = ckpt.get("metrics", {})

    print("=== Loaded metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
