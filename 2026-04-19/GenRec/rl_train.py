import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SyntheticGenRecDataset, collate_batch, GenRecVocab
from model import CausalTransformer


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.reshape(probs.shape[:-1])


def compute_reward(sampled: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Toy reward: fraction of target tokens matched (ignoring -100 mask).
    mask = labels != -100
    correct = (sampled == labels) & mask
    return correct.float().sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SyntheticGenRecDataset(num_samples=2000)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_batch)

    vocab_size = GenRecVocab.sid_base + 97
    model = CausalTransformer(vocab_size=vocab_size, embed_dim=256, num_layers=4, num_heads=8).to(device)
    try:
        model.load_state_dict(torch.load("genrec_sft.pt", map_location=device))
    except FileNotFoundError:
        pass

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    group_size = 4
    beta_nll = 0.1

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        sep_pos = (input_ids == GenRecVocab.sep_id).int().argmax(dim=1)
        prefix_len = int(sep_pos.max().item()) + 1

        logits = model(input_ids, attention_mask=attention_mask, prefix_len=prefix_len)

        token_logits = logits[:, :-1, :]
        target = input_ids[:, 1:]
        target_labels = labels[:, 1:]

        bsz, t, v = token_logits.shape
        token_logits = token_logits.reshape(bsz * t, v)
        target = target.reshape(bsz * t)
        target_labels = target_labels.reshape(bsz * t)

        logp_all = torch.log_softmax(token_logits, dim=-1)
        chosen_logp = logp_all[torch.arange(bsz * t, device=device), target]
        chosen_logp = chosen_logp.reshape(bsz, t)

        mask = (target_labels.reshape(bsz, t) != -100).float()
        seq_logp = (chosen_logp * mask).sum(dim=1)

        with torch.no_grad():
            rewards = []
            for _ in range(group_size):
                sampled_tokens = sample_with_temperature(logits, temperature=1.0)
                rewards.append(compute_reward(sampled_tokens, labels))
            rewards = torch.stack(rewards, dim=0)
            advantages = rewards.mean(dim=0) - rewards.mean(dim=0).mean()

        pg_loss = -(advantages.detach() * seq_logp).mean()
        nll_loss = F.cross_entropy(token_logits, target_labels, ignore_index=-100)
        loss = pg_loss + beta_nll * nll_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f} pg={pg_loss.item():.4f} nll={nll_loss.item():.4f}")
        if step >= 200:
            break

    torch.save(model.state_dict(), "genrec_rl.pt")
    print("saved genrec_rl.pt")


if __name__ == "__main__":
    main()
