import torch
from torch import nn
import torch.nn.functional as F


class UNIVIDLite(nn.Module):
    def __init__(self, frame_dim=32, text_dim=16, hidden=64, num_policies=5, caption_vocab=8):
        super().__init__()
        self.frame_encoder = nn.Sequential(nn.Linear(frame_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.text_encoder = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.policy_embed = nn.Embedding(num_policies, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.1))
        self.caption_head = nn.Linear(hidden, caption_vocab)
        self.violation_head = nn.Linear(hidden + caption_vocab, 1)
        self.leakage_head = nn.Linear(hidden + caption_vocab, 1)

    def forward(self, frame_features, text_features, policy_id):
        video_repr = self.frame_encoder(frame_features).mean(dim=1)
        text_repr = self.text_encoder(text_features)
        policy_repr = self.policy_embed(policy_id)
        fused = self.fusion(torch.cat([video_repr, text_repr, policy_repr], dim=-1))
        caption_logits = self.caption_head(fused)
        caption_evidence = torch.sigmoid(caption_logits)
        decision_repr = torch.cat([fused, caption_evidence], dim=-1)
        return {
            "caption_logits": caption_logits,
            "violation_logit": self.violation_head(decision_repr).squeeze(-1),
            "leakage_logit": self.leakage_head(decision_repr).squeeze(-1),
            "embedding": F.normalize(fused, dim=-1),
        }


class UNIVIDRAG(nn.Module):
    def __init__(self, lite_model, memory_size=64, hidden=64):
        super().__init__()
        self.lite_model = lite_model
        self.register_buffer("memory", F.normalize(torch.randn(memory_size, hidden), dim=-1))
        self.memory_labels = nn.Parameter(torch.zeros(memory_size), requires_grad=False)
        self.rag_head = nn.Linear(hidden + 1, 1)

    def forward(self, frame_features, text_features, policy_id):
        base = self.lite_model(frame_features, text_features, policy_id)
        sim = base["embedding"] @ self.memory.t()
        top_scores, top_idx = sim.topk(k=min(5, self.memory.shape[0]), dim=-1)
        retrieved_risk = self.memory_labels[top_idx].float().mean(dim=-1, keepdim=True)
        rag_logit = self.rag_head(torch.cat([base["embedding"], retrieved_risk], dim=-1)).squeeze(-1)
        base["rag_violation_logit"] = rag_logit
        base["retrieval_scores"] = top_scores
        return base


def loss_fn(outputs, batch):
    caption_loss = F.binary_cross_entropy_with_logits(outputs["caption_logits"], batch["caption_targets"])
    violation_loss = F.binary_cross_entropy_with_logits(outputs["violation_logit"], batch["violation_label"])
    leakage_loss = F.binary_cross_entropy_with_logits(outputs["leakage_logit"], batch["leakage_label"])
    return caption_loss + violation_loss + 0.5 * leakage_loss
