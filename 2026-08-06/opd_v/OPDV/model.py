import torch
from torch import nn
import torch.nn.functional as functional


class TinyMLLM(nn.Module):
    def __init__(self, image_dim=48, text_dim=32, hidden_dim=96, vocab_size=64, answer_len=8):
        super().__init__()
        self.answer_len = answer_len
        self.visual_encoder = nn.Linear(image_dim, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, answer_len * vocab_size))
        self.attention_probe = nn.Linear(hidden_dim * 2, 2)
        self.vocab_size = vocab_size

    def forward(self, image, text):
        visual_state = torch.tanh(self.visual_encoder(image))
        text_state = torch.tanh(self.text_encoder(text))
        fused = torch.cat([visual_state, text_state], dim=-1)
        logits = self.fusion(fused).view(image.size(0), self.answer_len, self.vocab_size)
        modality_attention = torch.softmax(self.attention_probe(fused), dim=-1)
        return logits, modality_attention


def modality_balance_ratio(attention):
    visual_attention = attention[:, 0].clamp_min(1e-6)
    text_attention = attention[:, 1].clamp_min(1e-6)
    return visual_attention / text_attention


def opd_v_loss(student_logits, positive_logits, negative_logits, labels, positive_attention, negative_attention, margin_threshold=0.05):
    task_loss = functional.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1))
    token_margin = functional.log_softmax(positive_logits, dim=-1) - functional.log_softmax(negative_logits, dim=-1)
    label_margin = torch.gather(token_margin, -1, labels.unsqueeze(-1)).squeeze(-1)
    trust_region = (label_margin > margin_threshold).float()
    teacher_distribution = torch.softmax(positive_logits.detach(), dim=-1)
    distillation = -(teacher_distribution * functional.log_softmax(student_logits, dim=-1)).sum(dim=-1)
    distillation_loss = (distillation * trust_region).sum() / trust_region.sum().clamp_min(1.0)
    balance_gap = (modality_balance_ratio(positive_attention) - modality_balance_ratio(negative_attention)).mean()
    balance_regularizer = -0.01 * balance_gap
    return task_loss + distillation_loss + balance_regularizer
