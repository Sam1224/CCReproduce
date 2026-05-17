from __future__ import annotations

import argparse

import torch

from cq_sid.data.toy import build_toy_corpus
from cq_sid.models.seq2seq import TinyTransformerSeq2Seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/eg_grpo.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    from cq_sid.utils.tokenizer import Vocab

    query_vocab = Vocab.from_state(ckpt["query_vocab"])
    target_vocab = Vocab.from_state(ckpt["target_vocab"])

    model = TinyTransformerSeq2Seq(
        src_vocab_size=len(query_vocab.id_to_token),
        tgt_vocab_size=len(target_vocab.id_to_token),
        d_model=192,
        nhead=4,
        num_layers=2,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    _items, _users, examples = build_toy_corpus()

    def encode_query(q: str):
        ids = [query_vocab.token_to_id["<bos>"]] + query_vocab.encode(q.lower().split()) + [
            query_vocab.token_to_id["<eos>"]
        ]
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(t, dtype=torch.bool)
        return t, mask

    for ex in examples[:5]:
        src_ids, src_mask = encode_query(ex.query)
        out = model.generate(
            src_ids,
            src_mask,
            bos_id=target_vocab.token_to_id["<bos>"],
            eos_id=target_vocab.token_to_id["<eos>"],
            pad_id=target_vocab.token_to_id["<pad>"],
            max_len=8,
            num_samples=1,
            temperature=0.9,
        )[0, 0]
        toks = target_vocab.decode(out.tolist())
        toks = [t for t in toks if t not in {"<pad>", "<bos>"}]
        if "<eos>" in toks:
            toks = toks[: toks.index("<eos>")]
        print("query=", ex.query)
        print("pred SID tokens=", toks)
        print("gt item=", ex.item_id)
        print("-")


if __name__ == "__main__":
    main()
