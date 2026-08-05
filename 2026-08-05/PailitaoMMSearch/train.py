"""train.py

训练脚本：
- stage1_pretrain: 持续预训练（toy 版对比学习 + VQ commitment loss）
- stage2_distill:  teacher 蒸馏（toy teacher + MSE）
- stage3_posttrain_hybrid_inference: 混合推理后训练（冻结 encoder，仅训练混合打分头）

输出：保存 checkpoint（包含 model state + tokenizer state + data config）。

运行：
    python train.py --ckpt_path checkpoints/pailitao_mmsearch.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SimpleTokenizer, ToyMMSearchDataset, build_toy_mmsearch_data, collate_fn
from model import HybSIDDualEncoder, ToyTeacherEncoder, margin_ranking_loss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def stage1_pretrain(
    model: HybSIDDualEncoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 2e-3,
    vq_weight: float = 0.2,
) -> None:
    """Stage1：持续预训练（toy 版）。"""

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"stage1 ep{ep}")
        for batch in pbar:
            batch = move_to_device(batch, device)

            q = model.encode(batch["query_input_ids"], batch["query_image"])
            i = model.encode(batch["pos_input_ids"], batch["pos_image"])

            loss_nce = model.contrastive_loss(q.z_hyb, i.z_hyb)
            loss = loss_nce + vq_weight * (q.vq_loss + i.vq_loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": float(loss.item()), "nce": float(loss_nce.item())})


def stage2_distill(
    model: HybSIDDualEncoder,
    teacher: ToyTeacherEncoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    distill_weight: float = 1.0,
    vq_weight: float = 0.1,
) -> None:
    """Stage2：蒸馏（toy teacher）。"""

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"stage2 ep{ep}")
        for batch in pbar:
            batch = move_to_device(batch, device)

            q = model.encode(batch["query_input_ids"], batch["query_image"])
            i = model.encode(batch["pos_input_ids"], batch["pos_image"])

            with torch.no_grad():
                tq = teacher(batch["query_input_ids"], batch["query_image"])
                ti = teacher(batch["pos_input_ids"], batch["pos_image"])

            loss_nce = model.contrastive_loss(q.z_hyb, i.z_hyb)
            loss_distill = torch.mean((q.z_cont - tq) ** 2) + torch.mean((i.z_cont - ti) ** 2)
            loss = loss_nce + distill_weight * loss_distill + vq_weight * (q.vq_loss + i.vq_loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": float(loss.item()), "nce": float(loss_nce.item()), "distill": float(loss_distill.item())})


def stage3_posttrain_hybrid_inference(
    model: HybSIDDualEncoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 5e-3,
) -> None:
    """Stage3：混合推理后训练（冻结 encoder，只训练 score_head）。

    我们用一个最小的 ranking loss：
    - 正样本：batch 内对应的 (q, pos)
    - 负样本：把 pos 在 batch 内循环平移一位，构造 (q, neg)

    真实系统中负样本构造会复杂得多（曝光/点击、hard negative mining、跨域等）。
    """

    # freeze except score_head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.score_head.parameters():
        p.requires_grad = True

    model.train()
    opt = torch.optim.Adam(model.score_head.parameters(), lr=lr)

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"stage3 ep{ep}")
        for batch in pbar:
            batch = move_to_device(batch, device)

            q = model.encode(batch["query_input_ids"], batch["query_image"])
            pos = model.encode(batch["pos_input_ids"], batch["pos_image"])

            # 构造一个简单的 in-batch negative
            neg_ids = torch.roll(batch["pos_input_ids"], shifts=1, dims=0)
            neg_img = torch.roll(batch["pos_image"], shifts=1, dims=0)
            neg = model.encode(neg_ids, neg_img)

            pos_score = model.score_head(q.z_cont, pos.z_cont, q.sid, pos.sid)
            neg_score = model.score_head(q.z_cont, neg.z_cont, q.sid, neg.sid)
            loss = margin_ranking_loss(pos_score, neg_score, margin=0.2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": float(loss.item()), "w_cont": float(model.score_head.w_cont.item()), "w_sid": float(model.score_head.w_sid.item())})

    # unfreeze for safety（后续如果继续训练）
    for p in model.parameters():
        p.requires_grad = True


def save_checkpoint(
    ckpt_path: str,
    model: HybSIDDualEncoder,
    tokenizer: SimpleTokenizer,
    config: Dict,
) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "config": config,
    }
    torch.save(payload, ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_items", type=int, default=200)
    parser.add_argument("--num_queries", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--num_codes", type=int, default=64)

    parser.add_argument("--epochs_stage1", type=int, default=2)
    parser.add_argument("--epochs_stage2", type=int, default=1)
    parser.add_argument("--epochs_stage3", type=int, default=1)

    parser.add_argument("--ckpt_path", type=str, default="checkpoints/pailitao_mmsearch.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog, train_q, _test_q, tok = build_toy_mmsearch_data(
        seed=args.seed,
        num_items=args.num_items,
        num_queries=args.num_queries,
        train_ratio=0.8,
    )

    train_ds = ToyMMSearchDataset(catalog=catalog, queries=train_q, tokenizer=tok, max_len=12)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = HybSIDDualEncoder(
        vocab_size=len(tok.vocab),
        d_model=args.d_model,
        num_groups=args.num_groups,
        num_codes=args.num_codes,
    ).to(device)

    teacher = ToyTeacherEncoder(vocab_size=len(tok.vocab), d_out=args.d_model).to(device)

    # stage1
    stage1_pretrain(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs_stage1,
    )

    # stage2
    stage2_distill(
        model=model,
        teacher=teacher,
        loader=train_loader,
        device=device,
        epochs=args.epochs_stage2,
    )

    # stage3
    stage3_posttrain_hybrid_inference(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs_stage3,
    )

    config = {
        "seed": args.seed,
        "num_items": args.num_items,
        "num_queries": args.num_queries,
        "d_model": args.d_model,
        "num_groups": args.num_groups,
        "num_codes": args.num_codes,
    }

    save_checkpoint(args.ckpt_path, model, tok, config)
    print(f"Saved checkpoint to: {args.ckpt_path}")


if __name__ == "__main__":
    main()
