"""Scalable soft graph clustering via differentiable soft modularity (Sec. 3.2).

Soft membership P = softmax(Z) in R^{|I| x C}. We maximize the closed-form
expected modularity (Proposition 3, Eq. 4):

    Q_soft(P) = (1/|E|) * sum_{(i,j) in E} p_i^T p_j  -  gamma * ||P^T k||_2^2 / |E|^2

by gradient ascent (minimize -Q_soft). Z is initialized via Leiden if
python-igraph/leidenalg are available, else KMeans on item embeddings, else
random (gated in try/except, graceful fallback).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _kmeans(X, C, iters=25, seed=0):
    """Tiny numpy KMeans -> hard labels (fallback init)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=C, replace=False)
    cen = X[idx].copy()
    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(iters):
        d = ((X[:, None, :] - cen[None]) ** 2).sum(-1)
        labels = d.argmin(1)
        for c in range(C):
            pts = X[labels == c]
            if len(pts):
                cen[c] = pts.mean(0)
    return labels


def init_logits(edge_index, X_items, num_items, C, seed=0, init_scale=4.0):
    """Initialize Z (|I| x C). Try Leiden -> KMeans -> random. Returns (Z0, C)."""
    labels, used = None, "random"
    try:                                                   # Leiden (gated import)
        import igraph as ig
        import leidenalg
        und = edge_index[:, :edge_index.shape[1] // 2].t().tolist()
        g = ig.Graph(n=num_items + 1, edges=und)
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                        seed=seed)
        lab = np.array(part.membership)[1:num_items + 1]   # drop PAD node 0
        uniq = np.unique(lab)
        remap = {u: k for k, u in enumerate(uniq)}
        labels = np.array([remap[x] for x in lab])
        C = len(uniq)
        used = "leiden"
    except Exception:
        try:
            labels = _kmeans(X_items, C, seed=seed)
            used = "kmeans"
        except Exception:
            labels = np.random.default_rng(seed).integers(0, C, size=num_items)
            used = "random"
    Z = torch.full((num_items, C), 0.0)
    Z[torch.arange(num_items), torch.from_numpy(labels).long()] = init_scale
    Z += 0.01 * torch.randn(num_items, C)
    print(f"[clustering] init via {used}, C={C}")
    return Z.requires_grad_(True), C


def soft_modularity(P, edge_index, deg, gamma):
    """Eq. 4: closed-form expected (soft) modularity Q_soft(P)."""
    E = edge_index.shape[1]                                # |E| (duplicate rep)
    pi = P[edge_index[0]]
    pj = P[edge_index[1]]
    intra = (pi * pj).sum(1).sum() / E                     # (1/|E|) sum p_i^T p_j
    Pk = (P * deg.unsqueeze(1)).sum(0)                     # P^T k  in R^C
    null = gamma * (Pk ** 2).sum() / (E ** 2)              # gamma ||P^T k||^2/|E|^2
    return intra - null


def soft_cluster(edge_index, deg, X_items, num_items, C=8, gamma=0.8,
                 steps=300, lr=0.1, seed=0):
    """Optimize P by gradient ascent on Q_soft. Returns (P[detached], C, Q)."""
    deg = deg[1:num_items + 1]                             # align to items 1..N
    edge_index = edge_index - 1                            # shift ids to 0-based
    Z, C = init_logits(edge_index + 1, X_items, num_items, C, seed)
    opt = torch.optim.Adam([Z], lr=lr)
    q = 0.0
    for t in range(steps):
        opt.zero_grad()
        P = F.softmax(Z, dim=1)
        q = soft_modularity(P, edge_index, deg, gamma)
        (-q).backward()
        opt.step()
    P = F.softmax(Z.detach(), dim=1)
    print(f"[clustering] final Q_soft={q.item():.4f}")
    return P, C, q.item()


if __name__ == "__main__":
    from data import build_toy_data
    from graph import build_co_engagement_graph
    X, seqs, n = build_toy_data()
    ei, deg = build_co_engagement_graph(seqs, n)
    P, C, q = soft_cluster(ei, deg, X[1:], n, C=8)
    print("P", P.shape, "C", C)
