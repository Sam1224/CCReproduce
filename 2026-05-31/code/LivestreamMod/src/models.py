"""
Dual-path content moderation models for LivestreamMod.

Paper: "Dynamic Content Moderation in Livestreams: Combining Supervised
       Classification with MLLM-Boosted Similarity Matching" (arXiv:2512.03553)

Two paths:
  1. ClassificationPath: MLP classifier for known violation classes
  2. SimilarityPath:     Encoder + cosine similarity against reference case bank
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationPath(nn.Module):
    """Supervised multi-class classifier for known violation types.
    Input: fused multimodal feature (text+audio+visual concatenated)
    Output: class logits for num_classes violation categories
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class ModalityEncoder(nn.Module):
    """Lightweight encoder for one modality. Used in similarity path."""
    def __init__(self, input_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class SimilarityPath(nn.Module):
    """Reference-based similarity path with MLLM-distilled encoder.
    Encodes text/audio/visual separately, then fuses into a single embedding
    for retrieval against a reference violation case bank.
    """
    def __init__(self, modal_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.text_enc = ModalityEncoder(modal_dim, embed_dim)
        self.audio_enc = ModalityEncoder(modal_dim, embed_dim)
        self.visual_enc = ModalityEncoder(modal_dim, embed_dim)
        # Learned fusion weights (simplified; paper uses cross-modal attention)
        self.fusion = nn.Linear(embed_dim * 3, embed_dim)

    def encode(self, text: torch.Tensor, audio: torch.Tensor,
               visual: torch.Tensor) -> torch.Tensor:
        te = self.text_enc(text)
        ae = self.audio_enc(audio)
        ve = self.visual_enc(visual)
        fused = torch.cat([te, ae, ve], dim=-1)
        return F.normalize(self.fusion(fused), dim=-1)

    def forward(self, text, audio, visual, ref_bank: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text, audio, visual: current content features [B, modal_dim]
            ref_bank: reference case embeddings [N_ref, embed_dim]
        Returns:
            max cosine similarity against reference bank [B]
        """
        query_emb = self.encode(text, audio, visual)       # [B, embed_dim]
        sims = torch.matmul(query_emb, ref_bank.T)         # [B, N_ref]
        max_sim, _ = sims.max(dim=-1)                      # [B]
        return max_sim


class HybridModerator(nn.Module):
    """Combined dual-path moderator: classification + similarity.
    Final score = α * cls_violation_prob + (1-α) * max_sim
    """
    def __init__(self, input_dim: int = 384, modal_dim: int = 128,
                 embed_dim: int = 64, num_classes: int = 5, alpha: float = 0.5):
        super().__init__()
        self.cls_path = ClassificationPath(input_dim, input_dim // 2, num_classes)
        self.sim_path = SimilarityPath(modal_dim, embed_dim)
        self.alpha = alpha

    def forward(self, fused, text, audio, visual, ref_bank=None):
        cls_logits = self.cls_path(fused)
        cls_prob = torch.softmax(cls_logits, dim=-1)[:, 1:].sum(dim=-1)  # P(any violation)
        if ref_bank is not None:
            sim_score = self.sim_path(text, audio, visual, ref_bank)
            hybrid = self.alpha * cls_prob + (1 - self.alpha) * sim_score
        else:
            hybrid = cls_prob
        return {"cls_logits": cls_logits, "cls_violation_prob": cls_prob,
                "hybrid_score": hybrid}
