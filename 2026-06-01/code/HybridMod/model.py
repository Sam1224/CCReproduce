"""
HybridMod: Hybrid Multimodal Livestream Content Moderation
Reproduction of: "Dynamic Content Moderation in Livestreams: Combining
Supervised Classification with MLLM-Boosted Similarity Matching" (KDD 2026)
arXiv: 2512.03553

Architecture:
  Pipeline A — Supervised multimodal classifier (distilled from MLLM teacher)
  Pipeline B — MLLM-boosted similarity matching against reference gallery
  Fusion     — Score-level decision fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalEncoder(nn.Module):
    """Encodes text + audio + visual features into a joint representation."""

    def __init__(self, text_dim: int, audio_dim: int, visual_dim: int, hidden_dim: int):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Cross-modal attention fusion (Eq. 3 in paper, approximated)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, audio_feat, visual_feat):
        # Project each modality
        t = self.text_proj(text_feat).unsqueeze(1)    # (B, 1, H)
        a = self.audio_proj(audio_feat).unsqueeze(1)  # (B, 1, H)
        v = self.visual_proj(visual_feat).unsqueeze(1) # (B, 1, H)

        tokens = torch.cat([t, a, v], dim=1)  # (B, 3, H)
        # Self-attended fusion
        fused, _ = self.cross_attn(tokens, tokens, tokens)
        fused = self.out_norm(fused.mean(dim=1))  # (B, H)
        return fused


class SupervisedClassifierPipeline(nn.Module):
    """
    Pipeline A: Supervised multimodal classifier for known violation categories.
    The student model is distilled offline from an MLLM teacher (see mllm_teacher.py).

    Paper §3.1: "classification pipeline achieves 67% recall at 80% precision"
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = MultimodalEncoder(text_dim, audio_dim, visual_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text_feat, audio_feat, visual_feat):
        h = self.encoder(text_feat, audio_feat, visual_feat)
        logits = self.classifier(h)
        return logits, h  # return both logits and embedding for distillation


class ReferenceGallery(nn.Module):
    """
    Stores embeddings of labelled violation reference samples.
    Used by Pipeline B for nearest-neighbour similarity search.

    Paper §3.2: gallery updated offline; supports new/emerging violation types.
    """

    def __init__(self, embedding_dim: int, max_gallery_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Non-parametric: stored as a buffer (updated offline)
        self.register_buffer(
            "gallery_embeddings", torch.zeros(max_gallery_size, embedding_dim)
        )
        self.register_buffer(
            "gallery_labels", torch.zeros(max_gallery_size, dtype=torch.long)
        )
        self.gallery_size = 0

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        n = embeddings.size(0)
        end = min(self.gallery_size + n, self.gallery_embeddings.size(0))
        actual_n = end - self.gallery_size
        self.gallery_embeddings[self.gallery_size : end] = embeddings[:actual_n]
        self.gallery_labels[self.gallery_size : end] = labels[:actual_n]
        self.gallery_size = end

    def retrieve_top_k(self, query: torch.Tensor, k: int = 5):
        """Cosine similarity retrieval."""
        if self.gallery_size == 0:
            return None, None
        gallery = F.normalize(
            self.gallery_embeddings[: self.gallery_size], dim=-1
        )
        query_norm = F.normalize(query, dim=-1)
        sim = torch.mm(query_norm, gallery.t())  # (B, G)
        top_k_sim, top_k_idx = sim.topk(min(k, self.gallery_size), dim=-1)
        top_k_labels = self.gallery_labels[top_k_idx]  # (B, k)
        return top_k_sim, top_k_labels


class SimilarityMatchingPipeline(nn.Module):
    """
    Pipeline B: MLLM-boosted similarity matching for novel / subtle violations.

    The MLLM provides rich semantic embeddings; a lightweight re-ranker
    (distilled from MLLM) performs similarity scoring at inference time.

    Paper §3.2: "similarity pipeline achieves 76% recall at 80% precision"
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = MultimodalEncoder(text_dim, audio_dim, visual_dim, hidden_dim)
        # Re-ranker (distilled from MLLM; lightweight MLP over similarity features)
        self.reranker = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),  # +1 for top-1 similarity score
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )
        self.gallery = ReferenceGallery(hidden_dim)

    def forward(self, text_feat, audio_feat, visual_feat):
        h = self.encoder(text_feat, audio_feat, visual_feat)
        top_k_sim, top_k_labels = self.gallery.retrieve_top_k(h, k=5)
        if top_k_sim is None:
            # No gallery entries: return zero logits
            B = h.size(0)
            return torch.zeros(B, self.reranker[-1].out_features, device=h.device), h

        # Aggregate: top-1 similarity score as an additional feature
        top1_sim = top_k_sim[:, :1]  # (B, 1)
        combined = torch.cat([h, top1_sim], dim=-1)
        logits = self.reranker(combined)
        return logits, h


class DecisionFusionLayer(nn.Module):
    """
    Combines scores from Pipeline A and Pipeline B.
    Paper uses a threshold-based OR fusion: flag if either pipeline fires.
    We implement a learnable weighted combination as a softer alternative.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # Learnable per-class weights for the two pipelines
        self.alpha = nn.Parameter(torch.tensor(0.5))  # weight for Pipeline A

    def forward(self, logits_a, logits_b):
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * logits_a + (1 - alpha) * logits_b
        return fused


class HybridMod(nn.Module):
    """
    Complete HybridMod model.

    Usage:
        model = HybridMod(text_dim=768, audio_dim=128, visual_dim=2048,
                          hidden_dim=512, num_classes=10)
        logits = model(text_feat, audio_feat, visual_feat)
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 128,
        visual_dim: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 10,
    ):
        super().__init__()
        self.classifier_pipeline = SupervisedClassifierPipeline(
            text_dim, audio_dim, visual_dim, hidden_dim, num_classes
        )
        self.similarity_pipeline = SimilarityMatchingPipeline(
            text_dim, audio_dim, visual_dim, hidden_dim, num_classes
        )
        self.fusion = DecisionFusionLayer(num_classes)

    def forward(self, text_feat, audio_feat, visual_feat):
        logits_a, emb_a = self.classifier_pipeline(text_feat, audio_feat, visual_feat)
        logits_b, emb_b = self.similarity_pipeline(text_feat, audio_feat, visual_feat)
        fused_logits = self.fusion(logits_a, logits_b)
        return {
            "logits": fused_logits,
            "logits_a": logits_a,
            "logits_b": logits_b,
            "emb_a": emb_a,
            "emb_b": emb_b,
        }
