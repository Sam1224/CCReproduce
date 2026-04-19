import torch
from torch.utils.data import DataLoader

from dataset import SyntheticGenRecDataset, collate_batch, GenRecVocab
from model import CausalTransformer


def hit_at_k(pred_items: list[int], gt_items: set[int], k: int) -> float:
    return 1.0 if any(x in gt_items for x in pred_items[:k]) else 0.0


def decode_items(tokens: torch.Tensor, sid_len: int) -> list[int]:
    # Inverse of encode_semantic_id for our base-97 scheme.
    base = 97
    out = []
    t = tokens.tolist()
    for i in range(0, len(t), sid_len):
        chunk = t[i : i + sid_len]
        if len(chunk) < sid_len:
            break
        if any(x in (GenRecVocab.eos_id, GenRecVocab.sep_id, GenRecVocab.bos_id, GenRecVocab.pad_id) for x in chunk):
            break
        x = 0
        mul = 1
        for c in chunk:
            x += (c - GenRecVocab.sid_base) * mul
            mul *= base
        out.append(x)
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SyntheticGenRecDataset(num_samples=400)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_batch)

    vocab_size = GenRecVocab.sid_base + 97
    model = CausalTransformer(vocab_size=vocab_size, embed_dim=256, num_layers=4, num_heads=8).to(device)

    ckpt = "genrec_rl.pt" if torch.cuda.is_available() else "genrec_sft.pt"
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    except FileNotFoundError:
        print("No checkpoint found; run train.py first")
        return

    model.eval()

    h1 = h5 = h10 = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            sep_pos = (input_ids == GenRecVocab.sep_id).int().argmax(dim=1)
            prefix_len = int(sep_pos.max().item()) + 1

            logits = model(input_ids, attention_mask=attention_mask, prefix_len=prefix_len)
            pred = logits.argmax(dim=-1)

            for i in range(pred.size(0)):
                gt_tokens = labels[i][labels[i] != -100]
                gt_items = set(decode_items(gt_tokens, sid_len=ds.sid_len))

                pred_tokens = pred[i][prefix_len:]
                pred_items = decode_items(pred_tokens, sid_len=ds.sid_len)

                h1 += hit_at_k(pred_items, gt_items, 1)
                h5 += hit_at_k(pred_items, gt_items, 5)
                h10 += hit_at_k(pred_items, gt_items, 10)
                n += 1

    print(f"Hit@1={h1/n:.3f} Hit@5={h5/n:.3f} Hit@10={h10/n:.3f}")


if __name__ == "__main__":
    main()
