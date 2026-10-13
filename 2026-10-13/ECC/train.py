import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.optim import Adam

from data import (
    build_preference_examples,
    build_sft_examples,
    generate_creatives,
    mine_ctr_sensitive_attributes,
    split_train_valid,
)
from model import Tokenizer, SuggestionSeq2Seq, build_preference_batch, build_sft_batch, cpo_loss


OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


def iterate_batches(items: List, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def train() -> Tuple[Path, Tokenizer]:
    records = generate_creatives()
    selected_attrs = mine_ctr_sensitive_attributes(records)

    sft_rows = build_sft_examples(records, selected_attrs)
    preference_rows = build_preference_examples(records, selected_attrs)

    sft_train, sft_valid = split_train_valid(sft_rows)
    pref_train, pref_valid = split_train_valid(preference_rows)

    texts = []
    for prompt, target in sft_rows:
        texts.append(prompt)
        texts.append(target)
    for prompt, chosen, shuffled, rejected in preference_rows:
        texts.extend([prompt, chosen, shuffled, rejected])

    tokenizer = Tokenizer(texts)
    model = SuggestionSeq2Seq(vocab_size=tokenizer.vocab_size)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {
        "selected_attrs": selected_attrs,
        "sft_train_size": len(sft_train),
        "pref_train_size": len(pref_train),
        "epochs": [],
    }

    for epoch in range(8):
        model.train()
        epoch_loss = 0.0
        for batch_rows in iterate_batches(sft_train, batch_size=16):
            src_ids, tgt_ids = build_sft_batch(batch_rows, tokenizer)
            logits = model(src_ids, tgt_ids[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history["epochs"].append({"stage": "sft", "epoch": epoch, "loss": epoch_loss})

    for epoch in range(6):
        model.train()
        epoch_loss = 0.0
        for batch_rows in iterate_batches(pref_train, batch_size=16):
            batch = build_preference_batch(batch_rows, tokenizer)
            loss = cpo_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history["epochs"].append({"stage": "cpo", "epoch": epoch, "loss": epoch_loss})

    checkpoint_path = OUTPUT_DIR / "ecc_toy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "token_to_id": tokenizer.token_to_id,
            "selected_attrs": selected_attrs,
            "valid_examples": sft_valid,
            "valid_preferences": pref_valid,
        },
        checkpoint_path,
    )

    with open(OUTPUT_DIR / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    return checkpoint_path, tokenizer


if __name__ == "__main__":
    path, tokenizer = train()
    print("Saved checkpoint to", path)
    print("Vocab size:", tokenizer.vocab_size)
