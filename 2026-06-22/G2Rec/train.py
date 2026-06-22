"""End-to-end G2Rec training (Sec. 3): build co-engagement graph -> soft cluster
-> interest profile tokens -> train generative recommender with L_item +
lambda * L_profile.  Runs on CPU in well under a minute on the toy dataset.
"""
import argparse
import torch

torch.set_num_threads(4)  # avoid CPU thread oversubscription on the toy data

from data import build_toy_data, make_loaders
from graph import build_co_engagement_graph, m_from_theorem
from clustering import soft_cluster
from model import G2Rec

CKPT = "g2rec_ckpt.pt"


def evaluate(model, loader, ks=(1, 5, 10), n_neg=99, seed=123):
    """Recall@K / NDCG@K / MRR over 1 positive + n_neg sampled negatives."""
    import numpy as np
    model.eval()
    rng = np.random.default_rng(seed)
    rec = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr, total = 0.0, 0
    V = model.num_items
    with torch.no_grad():
        for hist, lengths, target in loader:
            logits = model.score_last(hist, lengths)       # [B, vocab]
            for b in range(hist.shape[0]):
                pos = int(target[b])
                negs = rng.choice(V, size=n_neg + 20, replace=False) + 1
                negs = [int(x) for x in negs if x != pos][:n_neg]
                cand = torch.tensor([pos] + negs)
                scores = logits[b, cand]
                rank = int((scores > scores[0]).sum().item())  # 0-based rank of pos
                for k in ks:
                    if rank < k:
                        rec[k] += 1.0
                        ndcg[k] += 1.0 / np.log2(rank + 2)
                mrr += 1.0 / (rank + 1)
                total += 1
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = float(rec[k] / total)
        out[f"NDCG@{k}"] = float(ndcg[k] / total)
    out["MRR"] = float(mrr / total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # 1) toy data
    X_full, seqs, num_items = build_toy_data(seed=args.seed)
    train_loader, valid_loader, test_loader = make_loaders(seqs, num_items)
    avg_len = sum(len(s) for s in seqs) / len(seqs)

    # 2) sparsified co-engagement graph (Eq. 1-2; m via Theorem 2)
    m = max(int(avg_len), min(m_from_theorem(num_items, avg_len), 64))
    edge_index, deg = build_co_engagement_graph(seqs, num_items, m=m, seed=args.seed)
    print(f"[graph] m/user={m}, edges(dup)={edge_index.shape[1]}")

    # 3) soft graph clustering -> soft membership P (Eq. 4)
    P, C, q = soft_cluster(edge_index, deg, X_full[1:], num_items,
                           C=args.C, gamma=args.gamma, seed=args.seed)

    # 4) recommender with interest-profile tokens (Eq. 6-8)
    model = G2Rec(X_full, P, C, d_model=64, max_items=max(len(s) for s in seqs) + 2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 5) train with L = L_item + lambda * L_profile (Eq. 13)
    for ep in range(args.epochs):
        model.train()
        tot = ti = tp = 0.0
        for seqs_b, lengths in train_loader:
            opt.zero_grad()
            loss, li, lp = model.loss(seqs_b, lengths, lam=args.lam)
            loss.backward()
            opt.step()
            tot += loss.item(); ti += li.item(); tp += lp.item()
        nb = len(train_loader)
        if ep % 5 == 0 or ep == args.epochs - 1:
            vm = evaluate(model, valid_loader)
            print(f"epoch {ep:02d} | loss {tot/nb:.4f} (item {ti/nb:.4f} "
                  f"profile {tp/nb:.4f}) | val R@10 {vm['Recall@10']:.4f} "
                  f"NDCG@10 {vm['NDCG@10']:.4f} MRR {vm['MRR']:.4f}")

    torch.save({"state_dict": model.state_dict(), "X_full": X_full, "P": P,
                "C": C, "seqs": seqs, "num_items": num_items, "d_model": 64,
                "max_items": max(len(s) for s in seqs) + 2}, CKPT)
    print(f"[saved] {CKPT}")
    tm = evaluate(model, test_loader)
    print("test:", {k: round(v, 4) for k, v in tm.items()})


if __name__ == "__main__":
    main()
