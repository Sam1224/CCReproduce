"""
Dual-Path Content Moderation Model
Reproduces the architecture from arXiv 2512.03553 (KDD 2026).

System overview (§3.1):
  Path 1 — Supervised Classification:
    Trained on labeled violation examples.
    MLLM provides soft-label distillation signal.

  Path 2 — MLLM-Boosted Similarity Matching:
    Semantic + perceptual embedding of live content.
    kNN retrieval against violation case store.
    MLLM distills richer embedding representations.

  Decision Fusion:
    PASS if both paths score below threshold.
    REMOVE if either path confidently flags.
    REVIEW if ambiguous.

Input modalities: visual frames + audio transcript + text (captions/subtitles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# Violation categories (simplified, production has ~100+ fine-grained)
VIOLATION_CLASSES = [
    "compliant",
    "obscene_sexual",
    "violence_gore",
    "hate_speech",
    "misinformation",
    "product_fraud",
    "minor_exploitation",
]

NUM_CLASSES = len(VIOLATION_CLASSES)


class MultimodalEncoder(nn.Module):
    """
    Lightweight multimodal encoder for real-time inference.
    Processes: video frames (visual) + ASR transcript (text) + audio features.
    In production: based on pretrained backbone (CLIP + Whisper + BERT).
    """
    def __init__(self, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Visual stream: simplified 2D conv feature extractor
        self.visual_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Text stream: simple embedding + mean pool
        self.text_embed = nn.Embedding(32000, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        # Audio stream: mel-spectrogram → features
        self.audio_net = nn.Sequential(
            nn.Conv1d(80, hidden_dim, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        video_frames: Optional[torch.Tensor] = None,  # [B, C, H, W]
        text_ids: Optional[torch.Tensor] = None,       # [B, L]
        audio_mel: Optional[torch.Tensor] = None,      # [B, 80, T]
    ) -> torch.Tensor:
        parts = []

        if video_frames is not None:
            parts.append(self.visual_net(video_frames))
        else:
            B = (text_ids if text_ids is not None else audio_mel).size(0)
            parts.append(torch.zeros(B, self.hidden_dim, device=next(self.parameters()).device))

        if text_ids is not None:
            embeds = self.text_embed(text_ids).mean(dim=1)
            parts.append(self.text_proj(embeds))
        else:
            B = parts[0].size(0)
            parts.append(torch.zeros(B, self.hidden_dim, device=parts[0].device))

        if audio_mel is not None:
            parts.append(self.audio_net(audio_mel))
        else:
            B = parts[0].size(0)
            parts.append(torch.zeros(B, self.hidden_dim, device=parts[0].device))

        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused)  # [B, output_dim]


class Path1Classifier(nn.Module):
    """
    Path 1: Supervised violation classifier.
    Trained with CE loss on hard labels + KL distillation from MLLM soft labels.

    L_path1 = (1 - α) * CE(logits, hard_label) + α * KL(logits / T, mllm_soft / T)
    where T is temperature for distillation (typically 4.0), α = 0.5.
    """
    def __init__(self, input_dim: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)  # [B, num_classes]


class Path2SimilarityMatcher(nn.Module):
    """
    Path 2: MLLM-boosted similarity matching.
    Projects features into an embedding space where similarity to known
    violation cases indicates violation likelihood.

    L_path2 = contrastive_loss(anchor, positive_violation, negative_compliant)
    + MLLM embedding distillation loss (cosine alignment)
    """
    def __init__(self, input_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, embed_dim),
        )
        # L2 normalization for cosine similarity
        self.normalize = nn.functional.normalize

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeds = self.projector(features)
        return F.normalize(embeds, dim=-1)  # [B, embed_dim]

    def similarity_score(
        self,
        query_embeds: torch.Tensor,
        store_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and violation store embeddings.
        Returns max similarity score (nearest-neighbor distance).
        """
        # query_embeds: [B, D], store_embeds: [N, D]
        sim = torch.matmul(query_embeds, store_embeds.t())  # [B, N]
        return sim.max(dim=-1).values  # [B] — nearest neighbor score


class DualPathModerationModel(nn.Module):
    """
    Full dual-path content moderation model (arXiv 2512.03553).

    Architecture:
      Shared encoder → Path1 (classification) + Path2 (similarity)
      → Decision fusion

    Key innovation: MLLM soft labels are used for distillation in BOTH paths:
      - Path1: distill classification distribution from MLLM
      - Path2: align embedding space with MLLM embeddings (richer semantics)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        feature_dim: int = 256,
        embed_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        path1_threshold: float = 0.5,
        path2_threshold: float = 0.7,
        distill_temp: float = 4.0,
        distill_alpha: float = 0.5,
    ):
        super().__init__()
        self.encoder = MultimodalEncoder(hidden_dim, feature_dim)
        self.path1 = Path1Classifier(feature_dim, num_classes)
        self.path2 = Path2SimilarityMatcher(feature_dim, embed_dim)

        self.path1_threshold = path1_threshold
        self.path2_threshold = path2_threshold
        self.distill_temp = distill_temp
        self.distill_alpha = distill_alpha

    def forward(
        self,
        video_frames: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.encoder(video_frames, text_ids, audio_mel)
        class_logits = self.path1(features)
        sim_embeds = self.path2(features)
        return {
            "features": features,
            "class_logits": class_logits,
            "sim_embeds": sim_embeds,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_labels: torch.Tensor,
        mllm_soft_labels: Optional[torch.Tensor] = None,
        mllm_embeds: Optional[torch.Tensor] = None,
        violation_store_embeds: Optional[torch.Tensor] = None,
        violation_store_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined training loss.

        Path 1 loss: CE + KL distillation from MLLM
          L1 = (1-α) * CE(logits, y_hard) + α * KL(logits/T, p_mllm/T) * T²

        Path 2 loss: NT-Xent contrastive + MLLM embedding alignment
          L2 = NT-Xent(sim_embeds, store_embeds) + cosine_distill(sim_embeds, mllm_embeds)

        Total: L = L1 + λ * L2  (λ = 0.5 in paper)
        """
        class_logits = outputs["class_logits"]
        sim_embeds = outputs["sim_embeds"]

        # --- Path 1 Loss ---
        ce_loss = F.cross_entropy(class_logits, hard_labels)

        if mllm_soft_labels is not None:
            # KL divergence distillation
            student_log_probs = F.log_softmax(class_logits / self.distill_temp, dim=-1)
            teacher_probs = F.softmax(mllm_soft_labels / self.distill_temp, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.distill_temp ** 2)
            path1_loss = (1 - self.distill_alpha) * ce_loss + self.distill_alpha * kl_loss
        else:
            path1_loss = ce_loss
            kl_loss = torch.tensor(0.0)

        # --- Path 2 Loss ---
        if violation_store_embeds is not None and violation_store_labels is not None:
            path2_loss = self._nt_xent_loss(sim_embeds, violation_store_embeds, hard_labels, violation_store_labels)
        else:
            path2_loss = torch.tensor(0.0, device=class_logits.device)

        # MLLM embedding alignment
        if mllm_embeds is not None:
            mllm_embeds_norm = F.normalize(mllm_embeds, dim=-1)
            embed_align_loss = 1 - (sim_embeds * mllm_embeds_norm).sum(dim=-1).mean()
        else:
            embed_align_loss = torch.tensor(0.0, device=class_logits.device)

        total_loss = path1_loss + 0.5 * (path2_loss + embed_align_loss)

        return {
            "loss": total_loss,
            "path1_ce_loss": ce_loss,
            "path1_kl_loss": kl_loss,
            "path2_contrastive_loss": path2_loss,
            "embed_align_loss": embed_align_loss,
        }

    def _nt_xent_loss(
        self,
        query_embeds: torch.Tensor,
        store_embeds: torch.Tensor,
        query_labels: torch.Tensor,
        store_labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        NT-Xent contrastive loss for similarity path.
        Positives: (query, store_entry) with same violation class.
        Negatives: different violation classes.
        """
        sim_matrix = torch.matmul(query_embeds, store_embeds.t()) / temperature  # [B, N]
        # Positive mask: same label
        pos_mask = (query_labels.unsqueeze(1) == store_labels.unsqueeze(0)).float()
        # Avoid all-negative batches
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=query_embeds.device)
        loss = -torch.log(
            (torch.exp(sim_matrix) * pos_mask).sum(dim=-1) /
            (torch.exp(sim_matrix).sum(dim=-1) + 1e-8)
        ).mean()
        return loss

    def predict(
        self,
        video_frames: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
        violation_store: Optional["ViolationStore"] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: dual-path decision fusion.

        Decision logic (§3.4):
          path1_violation = path1_score > path1_threshold
          path2_violation = max_sim_score > path2_threshold
          decision = REMOVE if (path1_violation OR path2_violation)
                     REVIEW  if (path1_score > 0.3 OR path2_score > 0.5)
                     PASS    otherwise
        """
        with torch.no_grad():
            outputs = self.forward(video_frames, text_ids, audio_mel)
            class_logits = outputs["class_logits"]
            sim_embeds = outputs["sim_embeds"]

            path1_probs = F.softmax(class_logits, dim=-1)
            path1_violation_score = 1 - path1_probs[:, 0]  # 1 - P(compliant)
            path1_pred_class = class_logits.argmax(dim=-1)

            path2_score = torch.zeros(sim_embeds.size(0), device=sim_embeds.device)
            if violation_store is not None:
                store_embeds = violation_store.get_embeddings()
                if store_embeds is not None:
                    path2_score = self.path2.similarity_score(sim_embeds, store_embeds)

            # Decision fusion
            decisions = []
            for i in range(sim_embeds.size(0)):
                p1 = path1_violation_score[i].item()
                p2 = path2_score[i].item()
                if p1 > self.path1_threshold or p2 > self.path2_threshold:
                    decisions.append("REMOVE")
                elif p1 > 0.3 or p2 > 0.5:
                    decisions.append("REVIEW")
                else:
                    decisions.append("PASS")

        return {
            "decisions": decisions,
            "path1_scores": path1_violation_score,
            "path2_scores": path2_score,
            "predicted_classes": path1_pred_class,
        }
