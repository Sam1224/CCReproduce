import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from dataset import SidConfig, ToyConfig, build_examples, build_vocab, collate_sft, generate_toy_catalog, split_train_test
from model import RQVAE, RqVaeConfig, Seq2SeqSidModel, eg_grpo_loss


def build_item_text_embeddings(items, vocab, emb_dim: int = 64):
    # Simple bag-of-words embeddings with fixed random token vectors.
    g = torch.Generator().manual_seed(0)
    token_emb = torch.randn((len(vocab), emb_dim), generator=g)

    x = []
    cats = []
    for it in items:
        ids = torch.tensor(vocab.encode(it.title_tokens), dtype=torch.long)
        x.append(token_emb[ids].mean(dim=0))
        cats.append(it.category)

    return torch.stack(x, dim=0), torch.tensor(cats, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    toy = ToyConfig()
    sid = SidConfig()
    vocab, specials = build_vocab(toy=toy, sid=sid)

    items, users = generate_toy_catalog(toy)
    item_x, item_cat = build_item_text_embeddings(items, vocab=vocab, emb_dim=64)

    rqcfg = RqVaeConfig(
        n_categories=toy.n_categories,
        input_dim=64,
        latent_dim=64,
        k1=sid.k1,
        k2=sid.k2,
        k3=sid.k3,
    )
    rqvae = RQVAE(rqcfg).to(device)

    # Stage 1: Item -> SID (RQ-VAE training)
    rqvae.train()
    opt = torch.optim.Adam(rqvae.parameters(), lr=2e-3)
    for epoch in range(40):
        loss, info = rqvae(item_x.to(device), item_cat.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"[RQ-VAE] epoch={epoch+1} loss={loss.item():.4f} recon={info['recon_loss'].item():.4f} commit={info['commit_loss'].item():.4f}")

    rqvae.eval()
    with torch.no_grad():
        idx1, idx2, idx3 = rqvae.encode_to_indices(item_x.to(device), item_cat.to(device))

    sid_tokens_by_item = {}
    for it, i1, i2, i3 in zip(items, idx1.tolist(), idx2.tolist(), idx3.tolist()):
        sid_tokens_by_item[it.item_id] = rqvae.sid_tokens_from_indices(it.category, i1, i2, i3)

    # Stage 2/3: Query(+User) -> SID (SFT)
    examples = build_examples(items, users, toy=toy, with_user=True)
    train_ex, test_ex = split_train_test(examples)

    from dataset import SftDataset

    train_ds = SftDataset(train_ex, items, sid_tokens_by_item, vocab=vocab, specials=specials)
    test_ds = SftDataset(test_ex, items, sid_tokens_by_item, vocab=vocab, specials=specials)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=lambda b: collate_sft(b, specials["pad"]))

    s2s = Seq2SeqSidModel(vocab_size=len(vocab), pad_id=specials["pad"], d_model=128).to(device)
    opt = torch.optim.AdamW(s2s.parameters(), lr=5e-4)

    s2s.train()
    step = 0
    while step < 1200:
        for batch in train_loader:
            step += 1
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            loss = s2s(src, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"[SFT] step={step} loss={loss.item():.4f}")
            if step >= 1200:
                break

    # Stage 4: EG-GRPO alignment (toy)
    s2s.train()
    opt = torch.optim.AdamW(s2s.parameters(), lr=2e-4)

    rl_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=lambda b: collate_sft(b, specials["pad"]))
    step = 0
    while step < 120:
        for batch in rl_loader:
            step += 1
            src = batch["src"].to(device)
            gt = batch["tgt"].to(device)
            loss = eg_grpo_loss(
                model=s2s,
                rqvae=rqvae,
                vocab=vocab,
                src=src,
                gt_tgt=gt,
                bos_id=specials["bos"],
                eos_id=specials["eos"],
                group_size=6,
                max_len=5,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0:
                print(f"[EG-GRPO] step={step} loss={loss.item():.4f}")
            if step >= 120:
                break

    # Save
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(
        {
            "toy": asdict(toy),
            "sid": asdict(sid),
            "rqcfg": asdict(rqcfg),
            "vocab": vocab.id_to_token,
            "specials": specials,
            "rqvae": rqvae.state_dict(),
            "s2s": s2s.state_dict(),
            "sid_tokens_by_item": sid_tokens_by_item,
        },
        os.path.join(ckpt_dir, "toy_ckpt.pt"),
    )

    print(f"Saved checkpoint to {ckpt_dir}/toy_ckpt.pt")


if __name__ == "__main__":
    main()
