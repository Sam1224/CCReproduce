from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class HybridLivestreamModerationModel(nn.Module):
    def __init__(
        self,
        num_frames: int = 6,
        visual_dim: int = 32,
        audio_dim: int = 16,
        text_dim: int = 24,
        num_classes: int = 5,
        hidden_dim: int = 96,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.visual_encoder = MLPEncoder(num_frames * visual_dim, hidden_dim, embedding_dim)
        self.audio_encoder = MLPEncoder(audio_dim, hidden_dim // 2, embedding_dim // 2)
        self.text_encoder = MLPEncoder(text_dim, hidden_dim // 2, embedding_dim // 2)
        fusion_dim = embedding_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.query_proj = nn.Linear(hidden_dim, embedding_dim)
        self.reference_proj = nn.Linear(hidden_dim, embedding_dim)
        self.reranker = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_modalities(self, frames: torch.Tensor, audio: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = frames.shape[0]
        visual = self.visual_encoder(frames.reshape(batch_size, -1))
        audio_embed = self.audio_encoder(audio)
        text_embed = self.text_encoder(text)
        fused = self.fusion(torch.cat([visual, audio_embed, text_embed], dim=-1))
        return {
            "visual": visual,
            "audio": audio_embed,
            "text": text_embed,
            "fused": fused,
        }

    def encode_reference_bank(self, reference_bank: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encode_modalities(reference_bank["frames"], reference_bank["audio"], reference_bank["text"])
        return F.normalize(self.reference_proj(enc["fused"]), dim=-1)

    def forward(self, frames: torch.Tensor, audio: torch.Tensor, text: torch.Tensor, reference_bank: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(frames, audio, text)
        fused = encoded["fused"]
        classifier_logits = self.classifier(fused)
        query_embedding = F.normalize(self.query_proj(fused), dim=-1)
        reference_embeddings = self.encode_reference_bank(reference_bank)
        similarity_scores = query_embedding @ reference_embeddings.T
        expanded_query = query_embedding.unsqueeze(1).expand(-1, reference_embeddings.shape[0], -1)
        expanded_ref = reference_embeddings.unsqueeze(0).expand(query_embedding.shape[0], -1, -1)
        rerank_input = torch.cat([expanded_query, expanded_ref, expanded_query * expanded_ref], dim=-1)
        rerank_scores = self.reranker(rerank_input).squeeze(-1)
        return {
            "classifier_logits": classifier_logits,
            "student_hidden": fused,
            "query_embedding": query_embedding,
            "reference_embeddings": reference_embeddings,
            "similarity_scores": similarity_scores,
            "rerank_scores": rerank_scores,
        }


def _masked_logsumexp(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(masked, dim=-1)


def hybrid_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    reference_labels: torch.Tensor,
    kd_temperature: float = 2.0,
) -> Dict[str, torch.Tensor]:
    class_loss = F.cross_entropy(outputs["classifier_logits"], batch["label"])
    teacher_probs = F.softmax(batch["teacher_logits"] / kd_temperature, dim=-1)
    student_log_probs = F.log_softmax(outputs["classifier_logits"] / kd_temperature, dim=-1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (kd_temperature ** 2)
    hidden_loss = F.mse_loss(outputs["student_hidden"], batch["teacher_hidden"])

    positive_mask = batch["label"].unsqueeze(1) == reference_labels.unsqueeze(0)
    risky_mask = batch["label"] != 0
    if risky_mask.any():
        risky_scores = outputs["similarity_scores"][risky_mask]
        risky_positive = positive_mask[risky_mask]
        numerator = _masked_logsumexp(risky_scores, risky_positive)
        denominator = torch.logsumexp(risky_scores, dim=-1)
        retrieval_loss = -(numerator - denominator).mean()
        rerank_targets = risky_positive.float()
        rerank_loss = F.binary_cross_entropy_with_logits(outputs["rerank_scores"][risky_mask], rerank_targets)
    else:
        retrieval_loss = outputs["similarity_scores"].sum() * 0.0
        rerank_loss = outputs["rerank_scores"].sum() * 0.0

    total = class_loss + 0.35 * kd_loss + 0.1 * hidden_loss + 0.25 * retrieval_loss + 0.15 * rerank_loss
    return {
        "total": total,
        "classification": class_loss.detach(),
        "kd": kd_loss.detach(),
        "hidden": hidden_loss.detach(),
        "retrieval": retrieval_loss.detach(),
        "rerank": rerank_loss.detach(),
    }
