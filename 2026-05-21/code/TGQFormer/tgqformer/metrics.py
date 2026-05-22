"""Retrieval metrics for I2I evaluation."""
import torch


def recall_at_k(query_embs: torch.Tensor, gallery_embs: torch.Tensor,
                query_pids: torch.Tensor, gallery_pids: torch.Tensor,
                k: int = 10) -> float:
    """Recall@K: fraction of queries with at least one true positive in top-K."""
    sim = torch.matmul(query_embs, gallery_embs.T)  # (Nq, Ng)
    topk_idx = sim.topk(k + 1, dim=-1).indices      # +1 to skip self
    correct = 0
    for i in range(len(query_embs)):
        qpid = query_pids[i].item()
        retrieved = gallery_pids[topk_idx[i]].tolist()
        # Exclude self (same index, same pid), check if any other match
        matches = [p for j, p in enumerate(retrieved) if p == qpid and topk_idx[i][j] != i]
        if matches:
            correct += 1
    return correct / len(query_embs)


def hit_rate_at_k(query_embs: torch.Tensor, gallery_embs: torch.Tensor,
                  query_pids: torch.Tensor, gallery_pids: torch.Tensor,
                  k: int = 100) -> float:
    """Hit Rate@K (H@K): same as Recall@K for single positive per query."""
    return recall_at_k(query_embs, gallery_embs, query_pids, gallery_pids, k)
