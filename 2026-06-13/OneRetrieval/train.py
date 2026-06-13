import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from data import (
    ToyConfig,
    build_editability_suite,
    build_items,
    build_queries,
    build_vocab,
    evaluate_hr,
    evaluate_injection,
    set_seed,
)
from model import KAEEncoder, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/oneretrieval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--max_len", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = ToyConfig()
    items = build_items(cfg, seed=args.seed)
    vocab, slot_maps = build_vocab(cfg, items)

    data = build_queries(cfg, items, vocab=vocab, slot_maps=slot_maps, seed=args.seed, max_len=args.max_len)
    edit = build_editability_suite(
        cfg, items, vocab=vocab, slot_maps=slot_maps, seed=args.seed, max_len=args.max_len
    )

    slot_sizes = [len(slot_maps[cb]) for cb in cfg.codebooks]

    mcfg = ModelConfig(vocab_size=len(vocab.id_to_token), max_len=args.max_len, d_model=args.d_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KAEEncoder(mcfg, slot_sizes=slot_sizes, pad_id=vocab.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    x_train = torch.tensor(data["x_train"], dtype=torch.long, device=device)
    m_train = torch.tensor(data["m_train"], dtype=torch.long, device=device)
    y_train = torch.tensor(data["y_train"], dtype=torch.long, device=device)

    model.train()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(x_train.size(0), device=device)
        total = 0.0
        for i in range(0, perm.numel(), args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = model(x_train[idx], m_train[idx])
            loss = 0.0
            for s, lg in enumerate(logits):
                loss = loss + F.cross_entropy(lg, y_train[idx, s])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.detach())

        print(f"[OneRetrievalToy] epoch={ep:02d} loss={total:.3f}")

    # evaluation
    model.eval()
    x_test = torch.tensor(data["x_test"], dtype=torch.long, device=device)
    m_test = torch.tensor(data["m_test"], dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model.predict(x_test, m_test).cpu().numpy()

    hr10 = evaluate_hr(pred, data["y_test"], edit["item_code_ids"], topk=10)
    hr50 = evaluate_hr(pred, data["y_test"], edit["item_code_ids"], topk=50)

    # editability: no bypass (model has never seen the injected keyword)
    inj_x = torch.tensor(np.stack([q[0] for q in edit["injected_queries"]]), dtype=torch.long, device=device)
    inj_m = torch.tensor(np.stack([q[1] for q in edit["injected_queries"]]), dtype=torch.long, device=device)
    with torch.no_grad():
        pred_inj_no = model.predict(inj_x, inj_m).cpu().numpy()

    iar10_no, ihr10_no = evaluate_injection(
        pred_inj_no,
        patched_code_ids=edit["patched_code_ids"],
        target_ids=edit["target_ids"],
        topk=10,
    )

    # editability: serving bypass forces the reserved slot
    with torch.no_grad():
        pred_inj_yes = (
            model.predict(
                inj_x,
                inj_m,
                forced_slot=int(edit["injected_cb_idx"]),
                forced_token_id=int(edit["injected_token_id"]),
            )
            .cpu()
            .numpy()
        )

    iar10_yes, ihr10_yes = evaluate_injection(
        pred_inj_yes,
        patched_code_ids=edit["patched_code_ids"],
        target_ids=edit["target_ids"],
        topk=10,
    )

    print("\n=== Metrics (toy) ===")
    print(f"HR@10={hr10:.4f} HR@50={hr50:.4f}")
    print("--- Editability (Injected keyword) ---")
    print(f"No bypass:  IAR@10={iar10_no:.4f} IHR@10={ihr10_no:.4f}")
    print(f"With bypass: IAR@10={iar10_yes:.4f} IHR@10={ihr10_yes:.4f}")

    ckpt = {
        "toy_cfg": asdict(cfg),
        "model_cfg": asdict(mcfg),
        "vocab": {"id_to_token": vocab.id_to_token, "pad_id": vocab.pad_id, "unk_id": vocab.unk_id},
        "slot_maps": slot_maps,
        "items": items,
        "edit": {
            "target_ids": edit["target_ids"].tolist(),
            "injected_cb_idx": int(edit["injected_cb_idx"]),
            "injected_token_id": int(edit["injected_token_id"]),
            "item_code_ids": edit["item_code_ids"].tolist(),
            "patched_code_ids": edit["patched_code_ids"].tolist(),
            "injected_keyword": edit["injected_keyword"],
        },
        "state_dict": model.state_dict(),
        "metrics": {
            "hr10": hr10,
            "hr50": hr50,
            "iar10_no_bypass": iar10_no,
            "ihr10_no_bypass": ihr10_no,
            "iar10_with_bypass": iar10_yes,
            "ihr10_with_bypass": ihr10_yes,
        },
    }

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt["metrics"], f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {ckpt_path}")


if __name__ == "__main__":
    main()
