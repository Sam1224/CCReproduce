from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OneSearchV2ToyDataset, collate_onesearchv2
from model import OneModel, onesearch_v2_loss, rdrop_kl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = OneSearchV2ToyDataset(num_samples=256, vocab_size=5000, cot_len=8)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_onesearchv2)

    model = OneModel(vocab_size=5000, d_model=256, cot_len=8).to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Two stochastic passes (R-Drop / self-distillation).
            out1 = model(batch)
            out2 = model(batch)

            loss1, logs = onesearch_v2_loss(
                out1,
                answer_tgt=batch["answer_tgt"],
                answer_mask=batch["answer_mask"],
                cot_tgt=batch["cot_tgt"],
                reward=batch["reward"],
            )
            loss2, _ = onesearch_v2_loss(
                out2,
                answer_tgt=batch["answer_tgt"],
                answer_mask=batch["answer_mask"],
                cot_tgt=batch["cot_tgt"],
                reward=batch["reward"],
            )

            kl = rdrop_kl(out1.answer_logits, out2.answer_logits, mask=batch["answer_mask"])
            loss = 0.5 * (loss1 + loss2) + 0.5 * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} total={loss.item():.4f} "
                    f"lm={logs['lm_loss']:.4f} cot={logs['cot_loss']:.4f} rew={logs['reward_loss']:.4f} kl={kl.item():.4f}"
                )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
