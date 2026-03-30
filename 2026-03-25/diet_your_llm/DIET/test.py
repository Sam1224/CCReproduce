from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import ToyTaskDataset, build_vocab, collate_batch
from diet import DietConfig, apply_pruning_mask, majority_vote_keep_mask, profile_dim_importance, summarize_mask, task_keep_mask_from_importance
from model import TinyLMConfig, TinyTransformerLM


def eval_task(
    model: TinyTransformerLM,
    dl: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dl:
            logits, _ = model(input_ids=batch["input_ids"].to(device), attn_mask=batch["attn_mask"].to(device))
            pred = logits.argmax(dim=-1).cpu()
            tgt = batch["target_id"]
            correct += int((pred == tgt).sum().item())
            total += int(tgt.numel())
    return correct / max(1, total)


def load_base_model(device: torch.device) -> Tuple[TinyTransformerLM, List[str]]:
    ckpt = torch.load("checkpoints/base.pt", map_location=device)
    cfg = TinyLMConfig(**ckpt["cfg"])
    model = TinyTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, list(ckpt["vocab"])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_tokens = load_base_model(device)

    vocab = build_vocab()
    if vocab.tokens != vocab_tokens:
        raise RuntimeError("Vocab mismatch; re-run train.py")

    pad_id = vocab.stoi["<pad>"]
    diet_cfg = DietConfig(sparsity=0.2, samples_per_task=100)

    tasks = ["add", "parity", "compare"]

    eval_dls = {
        t: DataLoader(
            ToyTaskDataset(t, vocab=vocab, num_samples=256, seed=123),
            batch_size=64,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        )
        for t in tasks
    }

    print("== Baseline ==")
    for t in tasks:
        acc = eval_task(model, eval_dls[t], device)
        print(f"{t}: acc={acc:.3f}")

    # Profile per-task importance with only 100 samples per task (paper).
    task_keeps = []
    for t in tasks:
        prof_dl = DataLoader(
            ToyTaskDataset(t, vocab=vocab, num_samples=diet_cfg.samples_per_task, seed=999),
            batch_size=25,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        )
        imp = profile_dim_importance(model, prof_dl, device)
        keep = task_keep_mask_from_importance(imp, sparsity=diet_cfg.sparsity)
        task_keeps.append(keep)

    global_keep = majority_vote_keep_mask(task_keeps)
    keep_n, eff_sparsity = summarize_mask(global_keep)
    print(f"DIET global mask: keep={keep_n}/{global_keep.numel()} effective_sparsity={eff_sparsity:.2%}")

    apply_pruning_mask(model, global_keep)

    print("== DIET-pruned (masked) ==")
    for t in tasks:
        acc = eval_task(model, eval_dls[t], device)
        print(f"{t}: acc={acc:.3f}")

    print("\nNotes: This is a toy reproduction skeleton. The paper's DIET performs structured pruning on LLM weights; here we apply an equivalent dimension mask during forward passes to keep the core algorithm (activation profiling + majority vote) runnable.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
