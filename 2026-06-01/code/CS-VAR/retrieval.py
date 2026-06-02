"""
Cross-session retrieval module for CS-VAR.

Maintains a historical session database indexed by session embeddings.
At training/inference time, retrieves K most similar historical sessions
for a given current session (approximate nearest neighbour via FAISS or brute-force).

Paper §3.2: "retrieve cross-session behavioral evidence" — sessions with similar
behavioral patterns (e.g., same script, similar gift patterns) are retrieved.
"""
import torch
import torch.nn.functional as F
import numpy as np


class SessionDatabase:
    """
    In-memory session database for cross-session retrieval.
    Production systems would use FAISS / Milvus / Elasticsearch.
    """

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.embeddings: list[torch.Tensor] = []  # (N, H) accumulated
        self.risk_labels: list[int] = []
        self.session_ids: list[str] = []

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor, ids=None):
        """Add session embeddings to the database."""
        self.embeddings.append(embeddings.detach().cpu())
        self.risk_labels.extend(labels.tolist())
        if ids is not None:
            self.session_ids.extend(ids)

    def build_index(self):
        """Finalize the database (normalize embeddings)."""
        if self.embeddings:
            self._all_emb = F.normalize(
                torch.cat(self.embeddings, dim=0), dim=-1
            )  # (N, H)
            self._all_labels = torch.tensor(self.risk_labels)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, k: int = 5) -> tuple:
        """
        Retrieve top-k most similar historical sessions.
        Args:
            query: (B, H) — current session embeddings
            k: number of neighbours to retrieve
        Returns:
            retrieved_emb: (B, K, H)
            retrieved_labels: (B, K)
        """
        if not hasattr(self, "_all_emb") or len(self._all_emb) == 0:
            B = query.size(0)
            placeholder = torch.zeros(B, k, self.hidden_dim)
            placeholder_labels = torch.zeros(B, k, dtype=torch.long)
            return placeholder, placeholder_labels

        query_norm = F.normalize(query.cpu(), dim=-1)  # (B, H)
        sim = torch.mm(query_norm, self._all_emb.t())  # (B, N)

        actual_k = min(k, sim.size(1))
        top_k_sim, top_k_idx = sim.topk(actual_k, dim=-1)  # (B, k)

        retrieved_emb = self._all_emb[top_k_idx]        # (B, k, H)
        retrieved_labels = self._all_labels[top_k_idx]  # (B, k)

        # Pad if actual_k < k
        if actual_k < k:
            pad_e = torch.zeros(query.size(0), k - actual_k, self.hidden_dim)
            pad_l = torch.zeros(query.size(0), k - actual_k, dtype=torch.long)
            retrieved_emb = torch.cat([retrieved_emb, pad_e], dim=1)
            retrieved_labels = torch.cat([retrieved_labels, pad_l], dim=1)

        return retrieved_emb.to(query.device), retrieved_labels.to(query.device)
