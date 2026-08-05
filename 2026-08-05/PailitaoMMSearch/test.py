"""test.py

评估 toy 多模态检索：Recall@K。

推理形态（对应论文的“离散 + 连续”混合）：
1) 先用离散语义 ID (sid) 做候选过滤/粗排（这里是暴力计算匹配率后取 topM）
2) 再用混合打分头 score_head（连续 cos + 离散匹配率）做 rerank

注意：真实系统会用倒排索引/ANN/在线索引更新，这里都省略。

运行：
    python test.py --ckpt_path checkpoints/pailitao_mmsearch.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from tqdm import tqdm

from data import SimpleTokenizer, ToyMMSearchDataset, build_toy_mmsearch_data, collate_fn
from model import HybSIDDualEncoder


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def build_item_index(
    model: HybSIDDualEncoder,
    catalog_ds: ToyMMSearchDataset,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """对 catalog 中每个 item 预编码，形成检索索引。"""

    # catalog_ds 里每条样本对应一个 query；但 item 信息存在 catalog。
    # 我们直接遍历 catalog，构造一个 batch encoder。

    item_input_ids = []
    item_images = []
    item_ids = []
    for it in catalog_ds.catalog:
        item_input_ids.append(catalog_ds._encode_pad(it.title))
        item_images.append(it.image.float())
        item_ids.append(it.item_id)

    item_input_ids = torch.stack(item_input_ids, dim=0)
    item_images = torch.stack(item_images, dim=0)
    item_ids = torch.tensor(item_ids, dtype=torch.long)

    zs_cont: List[torch.Tensor] = []
    sids: List[torch.Tensor] = []

    for st in range(0, len(item_ids), batch_size):
        ed = min(len(item_ids), st + batch_size)
        out = model.encode(item_input_ids[st:ed].to(device), item_images[st:ed].to(device))
        zs_cont.append(out.z_cont.cpu())
        sids.append(out.sid.cpu())

    return {
        "item_ids": item_ids,
        "z_cont": torch.cat(zs_cont, dim=0),
        "sid": torch.cat(sids, dim=0),
    }


@torch.no_grad()
def recall_at_k(ranks: List[int], k: int) -> float:
    # rank: 1 表示 top1 命中
    hit = [1.0 if r <= k else 0.0 for r in ranks]
    return sum(hit) / max(1, len(hit))


@torch.no_grad()
def evaluate(
    model: HybSIDDualEncoder,
    test_ds: ToyMMSearchDataset,
    device: torch.device,
    candidate_m: int = 50,
    ks: List[int] = [1, 5, 10],
) -> Dict[int, float]:
    model.eval()

    index = build_item_index(model, test_ds, device=device)
    item_ids = index["item_ids"]  # [N]
    item_z = index["z_cont"].to(device)  # [N, D]
    item_sid = index["sid"].to(device)  # [N, G]

    ranks: List[int] = []

    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    for batch in tqdm(loader, desc="eval"):
        batch = move_to_device(batch, device)

        q = model.encode(batch["query_input_ids"], batch["query_image"])
        target = batch["target_item_id"]  # [B]

        # 1) SID 粗排：匹配组数越多越靠前（暴力计算）
        # match: [B, N, G] -> match_rate: [B, N]
        match = (q.sid.unsqueeze(1) == item_sid.unsqueeze(0)).float()
        w = torch.nn.functional.softplus(model.score_head.sid_group_weight).to(device)
        sid_score = (match * w.view(1, 1, -1)).sum(dim=-1) / (w.sum() + 1e-8)

        # 取 topM candidates
        cand_score, cand_idx = torch.topk(sid_score, k=min(candidate_m, sid_score.size(1)), dim=1)

        # 2) rerank：混合打分头（连续 cos + 离散 match）
        # 为了复用 score_head，我们把 candidate 子集 gather 出来
        # q.z_cont: [B, D]
        # cand_z: [B, M, D]
        cand_z = item_z[cand_idx]  # [B, M, D]
        cand_sid = item_sid[cand_idx]  # [B, M, G]

        # score_head 是 pairwise（[B]），这里做 broadcast
        qz = q.z_cont.unsqueeze(1).expand_as(cand_z)
        qsid = q.sid.unsqueeze(1).expand_as(cand_sid)

        s_cont = (torch.nn.functional.normalize(qz, dim=-1) * torch.nn.functional.normalize(cand_z, dim=-1)).sum(dim=-1)  # [B, M]
        s_sid = model.score_head.sid_similarity(qsid.reshape(-1, qsid.size(-1)), cand_sid.reshape(-1, cand_sid.size(-1))).view(qsid.size(0), qsid.size(1))
        score = model.score_head.w_cont * s_cont + model.score_head.w_sid * s_sid

        # rank target within candidates; 如果 target 不在候选里，则 rank=inf（视为 miss）
        sorted_score, sorted_pos = torch.sort(score, dim=1, descending=True)
        sorted_item_id = item_ids.to(device)[cand_idx].gather(1, sorted_pos)

        for b in range(sorted_item_id.size(0)):
            tgt = int(target[b].item())
            hits = (sorted_item_id[b] == tgt).nonzero(as_tuple=False)
            if hits.numel() == 0:
                ranks.append(10**9)
            else:
                ranks.append(int(hits[0].item()) + 1)  # 1-based

    return {k: recall_at_k(ranks, k) for k in ks}


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[HybSIDDualEncoder, SimpleTokenizer, Dict]:
    payload = torch.load(ckpt_path, map_location=device)
    tok = SimpleTokenizer.from_state_dict(payload["tokenizer"])
    cfg = payload["config"]

    model = HybSIDDualEncoder(
        vocab_size=len(tok.vocab),
        d_model=cfg["d_model"],
        num_groups=cfg["num_groups"],
        num_codes=cfg["num_codes"],
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, tok, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/pailitao_mmsearch.pt")
    parser.add_argument("--candidate_m", type=int, default=50)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_checkpoint(args.ckpt_path, device=device)

    # 用 ckpt 中的 data config 重新生成同分布 toy 数据（可复现）
    catalog, _train_q, test_q, _tok2 = build_toy_mmsearch_data(
        seed=cfg["seed"],
        num_items=cfg["num_items"],
        num_queries=cfg["num_queries"],
        train_ratio=0.8,
    )
    test_ds = ToyMMSearchDataset(catalog=catalog, queries=test_q, tokenizer=tok, max_len=12)

    metrics = evaluate(
        model=model,
        test_ds=test_ds,
        device=device,
        candidate_m=args.candidate_m,
        ks=args.ks,
    )

    print("Recall@K:")
    for k in sorted(metrics.keys()):
        print(f"  R@{k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
