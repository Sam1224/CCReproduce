import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ECommerceSequenceConfig, SyntheticECommerceDataset
from model import FullAttentionRanker, TM20KConfig
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description="Smoke-test TM20K student or an untrained baseline.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--merged-len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TM20KConfig(max_seq_len=args.max_seq_len, merged_len=args.merged_len)
    model = FullAttentionRanker(cfg, use_token_merge=True).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
    dataset = SyntheticECommerceDataset(
        size=args.dataset_size,
        config=ECommerceSequenceConfig(max_seq_len=args.max_seq_len, seed=20260811),
    )
    metrics = evaluate(model, DataLoader(dataset, batch_size=64), device)
    print(metrics)


if __name__ == "__main__":
    main()
