"""
Reference case bank and retrieval for the similarity path.

The similarity path retrieves the most similar known-violation cases from a
reference bank and uses max cosine similarity as the violation score.
This enables zero-shot generalization to emerging violation types by simply
adding new reference cases to the bank — no retraining needed.

Paper: Section 3.3 "Reference-Based Similarity Retrieval"
"""
import torch
import torch.nn.functional as F
from typing import List, Dict


class ReferenceCaseBank:
    """
    In-memory reference case bank for similarity-based violation detection.
    Production: use FAISS or Milvus for large-scale ANN retrieval.
    """
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.embeddings: List[torch.Tensor] = []
        self.metadata: List[Dict] = []

    def add(self, embedding: torch.Tensor, meta: Dict):
        """Add a reference violation case embedding to the bank."""
        emb = F.normalize(embedding.detach().cpu(), dim=-1)
        self.embeddings.append(emb)
        self.metadata.append(meta)

    def build_index(self) -> torch.Tensor:
        """Stack all embeddings into a matrix for batch retrieval."""
        if not self.embeddings:
            raise ValueError("Reference bank is empty")
        return torch.stack(self.embeddings, dim=0)  # [N_ref, embed_dim]

    def retrieve_top_k(self, query_emb: torch.Tensor,
                       k: int = 5) -> List[Dict]:
        """Retrieve top-k most similar reference cases."""
        bank = self.build_index()
        q = F.normalize(query_emb.unsqueeze(0), dim=-1)  # [1, embed_dim]
        sims = (q @ bank.T).squeeze(0)                   # [N_ref]
        topk_idx = sims.topk(k).indices
        return [self.metadata[i] for i in topk_idx.tolist()]


def build_reference_bank(sim_path_model, val_loader, violation_only: bool = True,
                          device: str = "cpu") -> ReferenceCaseBank:
    """Build reference bank from validation set violation cases."""
    bank = ReferenceCaseBank(embed_dim=sim_path_model.fusion.out_features)
    sim_path_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"]
            is_viol = batch["is_violation"]
            embs = sim_path_model.encode(text, audio, visual)  # [B, embed_dim]
            for i in range(len(labels)):
                if not violation_only or is_viol[i].item() > 0:
                    bank.add(embs[i], {"label": labels[i].item(),
                                       "is_violation": is_viol[i].item()})
    print(f"Reference bank built: {len(bank.embeddings)} cases")
    return bank
