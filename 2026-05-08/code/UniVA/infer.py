"""UniVA inference / eval -- value-guided personalized beam search."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from data import ToyAdRecDataset, ToyConfig
from model import CommercialSIDTokenizer, GARDecoder, PersonalizedTrie, UniVAConfig, value_beam_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("univa.infer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/univa.pt")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    toy = ToyConfig()
    eval_ds = ToyAdRecDataset(toy, "eval")
    cfg_dict = torch.load(args.ckpt, map_location=args.device)["cfg"]
    cfg = UniVAConfig(**cfg_dict)
    item_meta = eval_ds.item_value_meta()

    tokenizer = CommercialSIDTokenizer(cfg, item_meta)
    decoder = GARDecoder(cfg)
    state = torch.load(args.ckpt, map_location=args.device)
    tokenizer.load_state_dict(state["tokenizer"])
    decoder.load_state_dict(state["decoder"])
    tokenizer.to(args.device).eval()
    decoder.to(args.device).eval()

    # Build trie over the full catalogue from the (item_id, sid_code) pairs.
    all_items = torch.arange(cfg.num_items, device=args.device)
    sid_codes = tokenizer.encode(all_items)
    trie = PersonalizedTrie()
    trie.fit(sid_codes.cpu(), all_items.tolist())

    hits = 0
    for i in range(min(len(eval_ds), 64)):
        sample = eval_ds[i]
        history = sample["history"].unsqueeze(0).to(args.device)
        chosen = value_beam_search(decoder, tokenizer, history, trie, beam=args.top_k)
        if chosen and chosen[0] == int(sample["target"].item()):
            hits += 1
    logger.info("toy hit@1 over 64 eval rows = %d/64", hits)


if __name__ == "__main__":
    main()
