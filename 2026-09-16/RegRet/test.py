from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import SyntheticRegRetConfig, SyntheticRegRetDataset
from model import RegRetConfig, RegRetToyModel


@torch.no_grad()
def retrieval_metrics(query: torch.Tensor, text: torch.Tensor, attr: torch.Tensor, ks: tuple[int, ...] = (1, 5)):
    sims = query @ text.t()
    out = {}
    for k in ks:
        topk = sims.topk(k=k, dim=1).indices
        matched_attr = attr[topk]
        out[f"recall@{k}"] = float((matched_attr == attr.unsqueeze(1)).any(dim=1).float().mean().item())
    return out


def raw_baseline(region_feat: torch.Tensor, global_feat: torch.Tensor, text_feat: torch.Tensor, attr: torch.Tensor):
    region_metrics = retrieval_metrics(region_feat, text_feat, attr)
    global_metrics = retrieval_metrics(global_feat[:, : text_feat.size(-1)], text_feat, attr)
    return region_metrics, global_metrics


def main() -> None:
    here = Path(__file__).resolve().parent
    ckpt_path = here / "checkpoints" / "regret_toy.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    data_cfg = SyntheticRegRetConfig(**ckpt["data_cfg"])
    model_cfg = RegRetConfig(**ckpt["model_cfg"])

    test_ds = SyntheticRegRetDataset(1500, data_cfg, seed=123)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = RegRetToyModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_query = []
    all_text = []
    all_attr = []
    raw_region = []
    raw_global = []
    raw_text = []
    for batch in loader:
        all_query.append(model.encode_query(batch["region_feat"], batch["global_feat"]))
        all_text.append(model.encode_text(batch["text_feat"]))
        all_attr.append(batch["attr_id"])
        raw_region.append(batch["region_feat"][:, : model_cfg.text_dim])
        raw_global.append(batch["global_feat"])
        raw_text.append(batch["text_feat"])

    query = torch.cat(all_query, dim=0)
    text = torch.cat(all_text, dim=0)
    attr = torch.cat(all_attr, dim=0)
    region_feat = torch.cat(raw_region, dim=0)
    global_feat = torch.cat(raw_global, dim=0)
    text_feat = torch.cat(raw_text, dim=0)

    regret_metrics = retrieval_metrics(query, text, attr)
    region_metrics, global_metrics = raw_baseline(region_feat, global_feat, text_feat, attr)

    print("RegRet toy evaluation")
    print(f"region-aware recall@1={regret_metrics['recall@1']:.4f} recall@5={regret_metrics['recall@5']:.4f}")
    print(f"region-only baseline recall@1={region_metrics['recall@1']:.4f} recall@5={region_metrics['recall@5']:.4f}")
    print(f"global-only baseline recall@1={global_metrics['recall@1']:.4f} recall@5={global_metrics['recall@5']:.4f}")


if __name__ == "__main__":
    main()
