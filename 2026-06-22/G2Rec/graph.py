"""Item-item co-engagement graph + theoretical sparsification (Sec. 3.1).

Co-engagement set E* = union_u (I_u x I_u)  (Eq. 1).  Since |E*| can be O(M^2),
we sparsify by sampling, per user, m co-engagements from I_u x I_u WITH
replacement (Eq. 2). Theorem 2 shows |E| = O(M log M) preserves the Laplacian;
we expose m_from_theorem() implementing that bound.
"""
import math
import numpy as np
import torch


def m_from_theorem(num_items, avg_len, eps=0.5, delta=0.1):
    """Theorem 2 sample size per user: m = 2N(1/(3eps)+1/eps^2) log(2|I|/delta)."""
    N = max(avg_len, 2)
    return int(math.ceil(2 * N * (1.0 / (3 * eps) + 1.0 / (eps ** 2)) *
                         math.log(2 * num_items / delta)))


def build_co_engagement_graph(sequences, num_items, m=None, seed=0):
    """Return (edge_index[2,E_dup], degree[num_items+1]) of the sparsified graph.

    edge_index uses the duplicate undirected representation: both (i,j) and (j,i)
    are present (paper convention for modularity). Degrees are from the binary
    adjacency of the unique undirected edges.
    """
    rng = np.random.default_rng(seed)
    edge_set = set()
    for seq in sequences:
        items = np.array(seq)
        if len(items) < 2:
            continue
        k = m if m is not None else len(items)            # default m = |I_u|
        a = rng.choice(items, size=k, replace=True)        # sample I_u x I_u
        b = rng.choice(items, size=k, replace=True)
        for i, j in zip(a, b):
            if i == j:
                continue
            edge_set.add((int(min(i, j)), int(max(i, j))))  # undirected, dedup
    if not edge_set:
        edge_set.add((1, min(2, num_items)))
    und = np.array(sorted(edge_set), dtype=np.int64)        # [U,2] unique edges
    # duplicate representation (both directions)
    dup = np.concatenate([und, und[:, ::-1]], axis=0)
    edge_index = torch.from_numpy(dup.T.copy())             # [2, E_dup]
    deg = torch.zeros(num_items + 1, dtype=torch.float32)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    return edge_index, deg


if __name__ == "__main__":
    from data import build_toy_data
    X, seqs, n = build_toy_data()
    ei, deg = build_co_engagement_graph(seqs, n)
    print("edges (dup)", ei.shape[1], "m(theorem)", m_from_theorem(n, 10))
    print("nonzero-degree items", int((deg > 0).sum()))
