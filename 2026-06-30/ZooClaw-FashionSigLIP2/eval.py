from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import ToyFashionRetrievalDataset, Vocab, build_default_vocab, collate
from model import DualEncoder, DualEncoderConfig, wise_ft_interpolate


@dataclass
class EvalResult:
    r_at_1: float
    r_at_5: float
    r_at_10: float


@torch.no_grad()
def compute_embeddings(
    model: DualEncoder,
    ds: ToyFashionRetrievalDataset,
    vocab: Vocab,
    use_long_query: bool,
    batch_size: int = 128,
    device: str = "cpu",
):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate(b, vocab, max_len=model.cfg.max_len),
    )

    image_embs = []
    text_embs = []

    model.eval()
    for batch in loader:
        imgs = batch["images"].to(device)
        if use_long_query:
            ids = batch["long_ids"].to(device)
            attn = batch["long_attn"].to(device)
        else:
            ids = batch["short_ids"].to(device)
            attn = batch["short_attn"].to(device)

        zi = model.encode_image(imgs)
        zt = model.encode_text(ids, attn)
        image_embs.append(zi.cpu())
        text_embs.append(zt.cpu())

    return torch.cat(image_embs, dim=0), torch.cat(text_embs, dim=0)


def recall_at_k(sim: torch.Tensor, k: int) -> float:
    # sim: (N, N), higher is better; correct match is diagonal
    topk = sim.topk(k=k, dim=1).indices
    target = torch.arange(sim.size(0)).view(-1, 1)
    hit = (topk == target).any(dim=1).float().mean().item()
    return float(hit)


def evaluate(
    model: DualEncoder,
    ds: ToyFashionRetrievalDataset,
    vocab: Vocab,
    use_long_query: bool,
    device: str,
) -> EvalResult:
    zi, zt = compute_embeddings(model, ds, vocab, use_long_query=use_long_query, device=device)
    sim = zi @ zt.t()
    return EvalResult(
        r_at_1=recall_at_k(sim, 1),
        r_at_5=recall_at_k(sim, 5),
        r_at_10=recall_at_k(sim, 10),
    )


def load_ckpt(path: Path) -> tuple[DualEncoder, Vocab, dict]:
    obj = torch.load(path, map_location="cpu")
    cfg = DualEncoderConfig(**obj["cfg"])
    vocab = Vocab(obj["vocab"])
    model = DualEncoder(cfg)
    model.load_state_dict(obj["state_dict"])
    return model, vocab, obj


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(__file__).resolve().parent
    base_path = root / "checkpoints" / "base.pt"
    ft_path = root / "checkpoints" / "finetuned.pt"

    if not base_path.exists() or not ft_path.exists():
        raise RuntimeError(
            "Missing checkpoints. Run `python train.py` first to generate base.pt and finetuned.pt."
        )

    base, vocab, base_obj = load_ckpt(base_path)
    finetuned, _, ft_obj = load_ckpt(ft_path)

    base = base.to(device)
    finetuned = finetuned.to(device)

    id_ds = ToyFashionRetrievalDataset(num_items=800, seed=17, ood=False)
    ood_ds = ToyFashionRetrievalDataset(num_items=800, seed=17, ood=True)

    def pretty(r: EvalResult) -> str:
        return f"R@1={r.r_at_1:.3f}  R@5={r.r_at_5:.3f}  R@10={r.r_at_10:.3f}"

    print("== In-domain (short query) ==")
    print("Base     ", pretty(evaluate(base, id_ds, vocab, use_long_query=False, device=device)))
    print(
        "Finetuned",
        pretty(evaluate(finetuned, id_ds, vocab, use_long_query=False, device=device)),
    )

    print("\n== In-domain (long query) ==")
    print("Base     ", pretty(evaluate(base, id_ds, vocab, use_long_query=True, device=device)))
    print(
        "Finetuned",
        pretty(evaluate(finetuned, id_ds, vocab, use_long_query=True, device=device)),
    )

    print("\n== OOD (short query) ==")
    print("Base     ", pretty(evaluate(base, ood_ds, vocab, use_long_query=False, device=device)))
    print(
        "Finetuned",
        pretty(evaluate(finetuned, ood_ds, vocab, use_long_query=False, device=device)),
    )

    # WiSE-FT: interpolate weights to recover OOD robustness
    base_state = {k: v.detach().cpu() for k, v in base_obj["state_dict"].items()}
    ft_state = {k: v.detach().cpu() for k, v in ft_obj["state_dict"].items()}

    best_alpha = None
    best_r10 = None
    rows = []

    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        m = DualEncoder(base.cfg)
        m.load_state_dict(wise_ft_interpolate(base_state, ft_state, alpha=alpha))
        m = m.to(device)

        r = evaluate(m, ood_ds, vocab, use_long_query=False, device=device)
        rows.append({"alpha": alpha, "ood_r10": r.r_at_10})
        if best_r10 is None or r.r_at_10 > best_r10:
            best_r10 = r.r_at_10
            best_alpha = alpha

    print("\n== WiSE-FT sweep (OOD short query, R@10) ==")
    for row in rows:
        mark = "*" if row["alpha"] == best_alpha else " "
        print(f"{mark} alpha={row['alpha']:.1f}  OOD R@10={row['ood_r10']:.3f}")

    out = root / "checkpoints" / "wise_ft.json"
    out.write_text(json.dumps({"best_alpha": best_alpha, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nsaved sweep: {out}")


if __name__ == "__main__":
    main()
