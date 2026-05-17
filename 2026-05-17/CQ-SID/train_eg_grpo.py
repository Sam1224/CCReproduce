from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cq_sid.data.toy import (
    TextSeqDataset,
    build_click_reward_matrix,
    build_toy_corpus,
    build_vocab,
    collate_text_batch,
)
from cq_sid.models.seq2seq import TinyTransformerSeq2Seq
from cq_sid.utils.format_sid import SID
from cq_sid.utils.seed import set_seed


def sid_tokens_to_item_index(tokens: list[str]) -> int | None:
    # tokens: [Ck, Qi, Q0, Q0]
    if len(tokens) < 2 or not tokens[1].startswith("Q"):
        return None
    try:
        return int(tokens[1][1:])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=str, default="outputs/sft.pt")
    parser.add_argument("--out", type=str, default="outputs/eg_grpo.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.sft, map_location="cpu")

    from cq_sid.utils.tokenizer import Vocab

    query_vocab = Vocab.from_state(ckpt["query_vocab"])
    target_vocab = Vocab.from_state(ckpt["target_vocab"])

    items, _users, examples = build_toy_corpus()

    # ground-truth toy CQ-SID
    item_to_sid_tokens = {}
    for idx, it in enumerate(items):
        sid = SID(category_id=it.category_id, codes=(idx, 0, 0))
        item_to_sid_tokens[it.item_id] = sid.to_tokens()

    queries = [ex.query for ex in examples]
    user_profiles = [ex.user_profile for ex in examples]
    targets = [item_to_sid_tokens[ex.item_id] for ex in examples]

    dataset = TextSeqDataset(
        queries=queries,
        targets=targets,
        query_vocab=query_vocab,
        target_vocab=target_vocab,
        add_user_profile=ckpt.get("stage") == "user_query2sid",
        user_profiles=user_profiles,
    )
    pad_q = query_vocab.token_to_id["<pad>"]
    pad_t = target_vocab.token_to_id["<pad>"]

    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        collate_fn=lambda b: collate_text_batch(b, pad_q, pad_t),
    )

    model = TinyTransformerSeq2Seq(
        src_vocab_size=len(query_vocab.id_to_token),
        tgt_vocab_size=len(target_vocab.id_to_token),
        d_model=192,
        nhead=4,
        num_layers=2,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    reward_mat = build_click_reward_matrix(examples, [it.item_id for it in items])

    bos = target_vocab.token_to_id["<bos>"]
    eos = target_vocab.token_to_id["<eos>"]

    # Simplified EG-GRPO:
    # - for each query, sample N candidates + 1 injected ground truth
    # - compute per-sample reward (click = 1 if hits GT item)
    # - compute group-relative advantage
    # - apply REINFORCE loss on generated tokens

    for step in range(1, args.steps + 1):
        batch = next(iter(loader))
        src_ids = batch["query_ids"].to(device)
        src_mask = batch["query_mask"].to(device)
        gt_tgt_ids = batch["target_ids"].to(device)
        bsz = src_ids.size(0)

        # sample candidates
        samples = model.generate(
            src_ids,
            src_mask,
            bos_id=bos,
            eos_id=eos,
            pad_id=pad_t,
            max_len=gt_tgt_ids.size(1),
            num_samples=args.num_samples,
            temperature=args.temperature,
        )  # [B, S, T]

        # inject ground truth as "expert guidance" sample
        samples = torch.cat([samples, gt_tgt_ids[:, None, :]], dim=1)  # [B, S+1, T]

        # decode + rewards
        rewards = torch.zeros((bsz, samples.size(1)), device=device)
        for i in range(bsz):
            gt_item = examples[i].item_id
            gt_col = [it.item_id for it in items].index(gt_item)
            for j in range(samples.size(1)):
                toks = target_vocab.decode(samples[i, j].tolist())
                # strip specials/pads
                toks = [t for t in toks if t not in {"<pad>", "<bos>", "<eos>"}]
                item_idx = sid_tokens_to_item_index(toks)
                if item_idx is None or item_idx >= len(items):
                    r = 0.0
                else:
                    pred_item = items[item_idx].item_id
                    r = float(pred_item == gt_item)
                rewards[i, j] = r

        # group-relative advantage
        adv = rewards - rewards.mean(dim=1, keepdim=True)

        # log-prob of sampled tokens under current policy
        # We re-run forward with teacher forcing on the sampled sequences.
        # Note: this is a simplification (no stop-at-eos masking).
        flat_samples = samples.reshape(bsz * samples.size(1), -1)
        flat_adv = adv.reshape(-1)

        src_rep = src_ids.repeat_interleave(samples.size(1), dim=0)
        src_mask_rep = src_mask.repeat_interleave(samples.size(1), dim=0)

        tgt_in = flat_samples[:, :-1]
        tgt_out = flat_samples[:, 1:]
        tgt_mask = tgt_in != pad_t

        out = model(
            src_rep,
            src_mask_rep,
            torch.cat([flat_samples[:, :1], tgt_in], dim=1),
            torch.cat([torch.ones((tgt_in.size(0), 1), device=device, dtype=torch.bool), tgt_mask], dim=1),
            pad_id=pad_t,
        )

        logits = out.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * tgt_mask.float()
        seq_logp = token_logp.sum(dim=1)

        loss = -(seq_logp * flat_adv.detach()).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(
                f"step={step} loss={loss.item():.4f} reward_mean={rewards.mean().item():.3f}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sft": args.sft,
            "query_vocab": query_vocab.to_state(),
            "target_vocab": target_vocab.to_state(),
            "model": model.state_dict(),
        },
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
