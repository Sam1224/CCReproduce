import argparse
import os

import numpy as np
import torch

from data import evaluate_hr, evaluate_injection
from model import KAEEncoder, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="runs/oneretrieval")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    toy_cfg = ckpt["toy_cfg"]
    model_cfg = ckpt["model_cfg"]
    slot_maps = ckpt["slot_maps"]

    slot_sizes = [len(slot_maps[cb]) for cb in toy_cfg["codebooks"]]

    mcfg = ModelConfig(**model_cfg)
    model = KAEEncoder(mcfg, slot_sizes=slot_sizes, pad_id=ckpt["vocab"]["pad_id"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # rebuild item codes
    item_code_ids = np.asarray(ckpt["edit"]["item_code_ids"], dtype=np.int64)

    # sample a small synthetic test set from stored items
    rng = np.random.RandomState(0)
    n = 2000
    y = item_code_ids[rng.choice(item_code_ids.shape[0], size=n, replace=True)]

    # build trivial queries from codes (attribute tokens are also query tokens)
    vocab = ckpt["vocab"]["id_to_token"]
    token_to_id = {t: i for i, t in enumerate(vocab)}

    max_len = mcfg.max_len
    x = np.zeros((n, max_len), dtype=np.int64)
    m = np.zeros((n, max_len), dtype=np.int64)

    inv_maps = {cb: {v: k for k, v in slot_maps[cb].items()} for cb in toy_cfg["codebooks"]}

    for i in range(n):
        q = ["find", "need"]
        for s, cb in enumerate(toy_cfg["codebooks"]):
            q.append(inv_maps[cb][int(y[i, s])])

        ids = [token_to_id.get(t, 1) for t in q][:max_len]
        x[i, : len(ids)] = ids
        m[i, : len(ids)] = 1

    xt = torch.tensor(x, dtype=torch.long)
    mt = torch.tensor(m, dtype=torch.long)

    with torch.no_grad():
        pred = model.predict(xt, mt).cpu().numpy()

    hr = evaluate_hr(pred, y, item_code_ids, topk=args.topk)
    print(f"HR@{args.topk}={hr:.4f}")

    # editability (using stored edit suite)
    patched = np.asarray(ckpt["edit"]["patched_code_ids"], dtype=np.int64)
    target_ids = np.asarray(ckpt["edit"]["target_ids"], dtype=np.int64)

    inj_keyword = ckpt["edit"]["injected_keyword"]
    inj_slot = int(ckpt["edit"]["injected_cb_idx"])
    inj_token = int(ckpt["edit"]["injected_token_id"])

    # make injected queries by adding the injected keyword token
    inj_n = 400
    inj_x = x[:inj_n].copy()
    inj_m = m[:inj_n].copy()
    inj_tok_id = token_to_id.get(inj_keyword, 1)
    for i in range(inj_n):
        pos = int(inj_m[i].sum())
        if pos < max_len:
            inj_x[i, pos] = inj_tok_id
            inj_m[i, pos] = 1

    inj_xt = torch.tensor(inj_x, dtype=torch.long)
    inj_mt = torch.tensor(inj_m, dtype=torch.long)

    with torch.no_grad():
        pred_no = model.predict(inj_xt, inj_mt).cpu().numpy()
        pred_yes = model.predict(inj_xt, inj_mt, forced_slot=inj_slot, forced_token_id=inj_token).cpu().numpy()

    iar_no, ihr_no = evaluate_injection(pred_no, patched_code_ids=patched, target_ids=target_ids, topk=args.topk)
    iar_yes, ihr_yes = evaluate_injection(pred_yes, patched_code_ids=patched, target_ids=target_ids, topk=args.topk)

    print(f"IAR@{args.topk} (no bypass)={iar_no:.4f}  IHR@{args.topk} (no bypass)={ihr_no:.4f}")
    print(f"IAR@{args.topk} (bypass)   ={iar_yes:.4f}  IHR@{args.topk} (bypass)   ={ihr_yes:.4f}")


if __name__ == "__main__":
    main()
