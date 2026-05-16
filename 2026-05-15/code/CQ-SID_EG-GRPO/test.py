import os

import torch
from torch.utils.data import DataLoader

from dataset import SftDataset, ToyConfig, build_examples, build_vocab, collate_sft, generate_toy_catalog, split_train_test
from model import RQVAE, RqVaeConfig, Seq2SeqSidModel, decode_sid_tokens


def greedy_generate(model: Seq2SeqSidModel, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 5):
    model.eval()
    device = src.device
    B = src.shape[0]

    src_key_padding_mask = src.eq(model.pad_id)
    memory = model.encoder(model._add_pos(src), src_key_padding_mask=src_key_padding_mask)

    seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        T = seq.shape[1]
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        dec = model.decoder(
            model._add_pos(seq),
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = model.lm_head(dec[:, -1, :])
        nxt = logits.argmax(dim=-1)
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        if (nxt == eos_id).all():
            break
    return seq


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "toy_ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Run train.py first. Missing: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    toy = ToyConfig(**ckpt["toy"])
    sid_tokens_by_item = ckpt["sid_tokens_by_item"]

    from dataset import SidConfig

    sid = SidConfig(**ckpt["sid"])
    vocab, specials = build_vocab(toy=toy, sid=sid)

    items, users = generate_toy_catalog(toy)
    examples = build_examples(items, users, toy=toy, with_user=True)
    _, test_ex = split_train_test(examples)

    test_ds = SftDataset(test_ex, items, sid_tokens_by_item, vocab=vocab, specials=specials)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=lambda b: collate_sft(b, specials["pad"]))

    rqcfg = RqVaeConfig(**ckpt["rqcfg"])
    rqvae = RQVAE(rqcfg)
    rqvae.load_state_dict(ckpt["rqvae"])

    s2s = Seq2SeqSidModel(vocab_size=len(vocab), pad_id=specials["pad"], d_model=128)
    s2s.load_state_dict(ckpt["s2s"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2s.to(device)

    total = 0
    exact = 0

    for batch in loader:
        src = batch["src"].to(device)
        gt = batch["tgt"].to(device)

        pred = greedy_generate(s2s, src, bos_id=specials["bos"], eos_id=specials["eos"], max_len=5)

        for p, g in zip(pred, gt):
            p_tokens = decode_sid_tokens(vocab, p)[:3]
            g_tokens = decode_sid_tokens(vocab, g)[:3]
            total += 1
            if p_tokens == g_tokens:
                exact += 1

    print(f"Toy Exact-SID Acc: {exact/total:.3f} ({exact}/{total})")


if __name__ == "__main__":
    main()
