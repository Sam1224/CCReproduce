"""
AIR (Atomic Intent Reasoning) — Model Architecture

Key modules (faithful to the paper):
  1. AtomicIntentIndex  — offline FAISS index of atom prototype vectors
  2. OnlineIntentComposer — lightweight ANN retrieval + linear composition
  3. CrossDomainUserEncoder — maps composed intent into commerce-side features
  4. AIRRankingModel — full ranking model integrating composed intent

Paper formula references:
  User intent composition:   u_intent = Σ_k w_k * a_k     (Eq. 1 in paper)
  Semantic consistency loss: L_sc = -cos(u_intent, u_online)  (Eq. 2)
  Ranking loss:              L_rank = BCE(score, label)       (Eq. 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from typing import Optional, Tuple


# ── Atomic Intent Index (offline, FAISS) ───────────────────────────────────

class AtomicIntentIndex:
    """
    Stores atom prototype vectors in a FAISS index for fast ANN retrieval.
    In production: populated offline by LLM reasoning over user behaviors.
    """

    def __init__(self, atom_dim: int = 128):
        self.atom_dim = atom_dim
        self.index = faiss.IndexFlatIP(atom_dim)   # inner-product search
        self.atom_vectors: Optional[np.ndarray] = None

    def build(self, atom_prototypes: np.ndarray):
        """Load atom prototype vectors into FAISS index."""
        assert atom_prototypes.shape[1] == self.atom_dim
        protos = atom_prototypes.astype(np.float32)
        # L2-normalize for cosine similarity via inner-product
        faiss.normalize_L2(protos)
        self.index.add(protos)
        self.atom_vectors = protos
        print(f"[AtomicIntentIndex] built with {self.index.ntotal} atoms")

    def retrieve(self, query: np.ndarray, top_k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            query: (B, atom_dim) float32
        Returns:
            scores: (B, top_k)
            indices: (B, top_k)
        """
        q = query.astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        return scores, indices


# ── Online Intent Composer ────────────────────────────────────────────────

class OnlineIntentComposer(nn.Module):
    """
    Given a user's online context (current session features), retrieves the
    most relevant atom vectors and composes a user intent representation.

    u_intent = Σ_k softmax(sim(q, a_k)) * a_k      (Eq. 1)
    """

    def __init__(self, atom_prototypes: torch.Tensor, top_k: int = 8, intent_dim: int = 128):
        super().__init__()
        self.top_k = top_k
        self.intent_dim = intent_dim
        # Register atom prototypes as a non-trainable buffer
        # (in real system: populated offline; here we allow fine-tuning for sim)
        self.register_buffer("atoms", F.normalize(atom_prototypes, dim=-1))

    def forward(self, session_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            session_emb: (B, intent_dim) — encoded current session
        Returns:
            u_intent: (B, intent_dim)
        """
        q = F.normalize(session_emb, dim=-1)              # (B, D)
        # Attention over all atoms (efficient for small n_atoms; use FAISS for large)
        sim = q @ self.atoms.T                             # (B, n_atoms)
        # Sparse top-k selection
        topk_vals, topk_idx = sim.topk(self.top_k, dim=-1)   # (B, k)
        weights = F.softmax(topk_vals, dim=-1)             # (B, k)
        # Gather selected atom vectors
        selected_atoms = self.atoms[topk_idx]              # (B, k, D)
        # Weighted sum
        u_intent = (weights.unsqueeze(-1) * selected_atoms).sum(dim=1)  # (B, D)
        return u_intent


# ── Session Encoder (content domain) ────────────────────────────────────────

class SessionEncoder(nn.Module):
    """Encodes content-domain item sequence into a session embedding."""

    def __init__(self, n_items: int, embed_dim: int = 64, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 20):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq: (B, L) item IDs
        Returns:
            session_emb: (B, hidden_dim)
        """
        B, L = item_seq.shape
        pos = torch.arange(L, device=item_seq.device).unsqueeze(0)
        x = self.item_emb(item_seq) + self.pos_emb(pos)
        # Mask padded positions (item_id == 0)
        key_padding_mask = (item_seq == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        # Use [CLS]-style: mean over non-padded positions
        mask = (~key_padding_mask).float().unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.proj(x)


# ── Cross-Domain Transfer Layer ───────────────────────────────────────────

class CrossDomainTransfer(nn.Module):
    """Maps content-side intent into commerce-side feature space."""

    def __init__(self, intent_dim: int = 128, commerce_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(intent_dim, intent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(intent_dim * 2),
            nn.Linear(intent_dim * 2, commerce_dim),
        )

    def forward(self, u_intent: torch.Tensor) -> torch.Tensor:
        return self.mlp(u_intent)


# ── Commerce Item Encoder ──────────────────────────────────────────────────

class CommerceItemEncoder(nn.Module):
    def __init__(self, n_items: int, embed_dim: int = 128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embed_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item_ids)


# ── Full AIR Ranking Model ────────────────────────────────────────────────

class AIRRankingModel(nn.Module):
    """
    Full AIR model integrating:
      - Content-domain session encoder
      - Online atomic intent composer
      - Cross-domain transfer
      - Commerce-side ranking

    Losses:
      L_total = L_rank + λ * L_sc
      L_sc = -cosine_similarity(u_intent_content, u_intent_commerce)  [Eq. 2]
      L_rank = BCE(dot(u_commerce, v_item), label)                    [Eq. 3]
    """

    def __init__(
        self,
        n_content_items: int,
        n_commerce_items: int,
        atom_prototypes: torch.Tensor,
        embed_dim: int = 64,
        intent_dim: int = 128,
        top_k_atoms: int = 8,
        sc_lambda: float = 0.1,
    ):
        super().__init__()
        self.sc_lambda = sc_lambda

        # Content session encoder
        self.content_encoder = SessionEncoder(
            n_items=n_content_items, embed_dim=embed_dim, hidden_dim=intent_dim
        )

        # Online intent composer
        self.composer = OnlineIntentComposer(
            atom_prototypes=atom_prototypes, top_k=top_k_atoms, intent_dim=intent_dim
        )

        # Cross-domain transfer
        self.transfer = CrossDomainTransfer(intent_dim=intent_dim, commerce_dim=intent_dim)

        # Commerce session encoder
        self.commerce_encoder = SessionEncoder(
            n_items=n_commerce_items, embed_dim=embed_dim, hidden_dim=intent_dim
        )

        # Commerce item encoder
        self.item_encoder = CommerceItemEncoder(n_items=n_commerce_items, embed_dim=intent_dim)

        # Final user representation MLP (fuse transferred + commerce encodings)
        self.user_mlp = nn.Sequential(
            nn.Linear(intent_dim * 2, intent_dim),
            nn.GELU(),
            nn.Linear(intent_dim, intent_dim),
        )

    def encode_user(
        self,
        content_seq: torch.Tensor,
        commerce_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            u_final: (B, intent_dim)  — final user representation
            u_intent: (B, intent_dim) — composed intent (for consistency loss)
        """
        # Content-side encoding
        session_emb = self.content_encoder(content_seq)          # (B, D)
        u_intent = self.composer(session_emb)                     # (B, D)  Eq. 1
        u_transferred = self.transfer(u_intent)                   # (B, D)

        # Commerce-side encoding
        u_commerce = self.commerce_encoder(commerce_seq)          # (B, D)

        # Fuse
        u_final = self.user_mlp(
            torch.cat([u_transferred, u_commerce], dim=-1)
        )                                                          # (B, D)
        return u_final, u_intent

    def forward(
        self,
        content_seq: torch.Tensor,
        commerce_seq: torch.Tensor,
        pos_item: torch.Tensor,
        neg_item: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            content_seq:  (B, L_c)
            commerce_seq: (B, L_e)
            pos_item:     (B,)  positive commerce item
            neg_item:     (B,)  negative commerce item (optional; sampled if None)

        Returns:
            dict with 'loss', 'pos_score', 'neg_score'
        """
        u_final, u_intent = self.encode_user(content_seq, commerce_seq)

        # Item representations
        v_pos = self.item_encoder(pos_item)                       # (B, D)
        if neg_item is None:
            # random negative
            neg_item = torch.randint(
                0, self.item_encoder.item_emb.num_embeddings,
                (pos_item.shape[0],), device=pos_item.device
            )
        v_neg = self.item_encoder(neg_item)

        # Ranking scores
        pos_score = (u_final * v_pos).sum(-1)                     # (B,)
        neg_score = (u_final * v_neg).sum(-1)                     # (B,)

        # BPR ranking loss  (Eq. 3 simplified as BPR)
        L_rank = -F.logsigmoid(pos_score - neg_score).mean()

        # Semantic consistency loss  (Eq. 2)
        # u_intent (content) vs u_final (commerce) — encourage alignment
        L_sc = 1 - F.cosine_similarity(u_intent, u_final, dim=-1).mean()

        loss = L_rank + self.sc_lambda * L_sc

        return {
            "loss": loss,
            "L_rank": L_rank.item(),
            "L_sc": L_sc.item(),
            "pos_score": pos_score.detach(),
            "neg_score": neg_score.detach(),
        }

    @torch.no_grad()
    def recommend(
        self,
        content_seq: torch.Tensor,
        commerce_seq: torch.Tensor,
        all_item_embs: torch.Tensor,
        top_k: int = 10,
    ) -> torch.Tensor:
        """Return top-k commerce item indices for each user in batch."""
        u_final, _ = self.encode_user(content_seq, commerce_seq)
        scores = u_final @ all_item_embs.T                        # (B, n_items)
        return scores.topk(top_k, dim=-1).indices


if __name__ == "__main__":
    B, L, D = 4, 20, 128
    n_content, n_commerce, n_atoms = 5000, 3000, 128

    atom_proto = torch.randn(n_atoms, D)
    model = AIRRankingModel(
        n_content_items=n_content,
        n_commerce_items=n_commerce,
        atom_prototypes=atom_proto,
        embed_dim=64,
        intent_dim=D,
    )
    content_seq = torch.randint(1, n_content, (B, L))
    commerce_seq = torch.randint(0, n_commerce, (B, L))
    pos_item = torch.randint(0, n_commerce, (B,))

    out = model(content_seq, commerce_seq, pos_item)
    print(f"Loss: {out['loss'].item():.4f} | L_rank: {out['L_rank']:.4f} | L_sc: {out['L_sc']:.4f}")

    all_items = model.item_encoder.item_emb.weight.detach()
    recs = model.recommend(content_seq, commerce_seq, all_items)
    print(f"Top-10 recommendations shape: {recs.shape}")
    print("Model OK.")
