import torch
from torch import nn
import torch.nn.functional as F


class AttributeValueExtractor(nn.Module):
    def __init__(self, input_dim=256, hidden=96, num_values=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_values),
        )

    def forward(self, features):
        return self.net(features)


class ArenaJudge(nn.Module):
    def __init__(self, input_dim=256, num_values=8, bias_strength=0.08, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("projection", torch.randn(input_dim, num_values, generator=gen) * 0.08)
        self.register_buffer("bias", torch.randn(num_values, generator=gen) * bias_strength)

    def forward(self, features, candidate_labels):
        logits = features @ self.projection + self.bias
        model_vote = logits.argmax(dim=-1)
        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
        return torch.where(confidence > 0.34, model_vote, candidate_labels)


class LLMArena:
    def __init__(self, num_judges=21, input_dim=256, num_values=8):
        self.judges = [ArenaJudge(input_dim, num_values, seed=i) for i in range(num_judges)]
        self.num_values = num_values

    def vote(self, features, candidate_labels):
        votes = torch.stack([judge(features, candidate_labels) for judge in self.judges], dim=0)
        one_hot = F.one_hot(votes, num_classes=self.num_values).float().sum(dim=0)
        majority = one_hot.argmax(dim=-1)
        agreement = one_hot.max(dim=-1).values / len(self.judges)
        return majority, agreement, votes


def cleaning_metrics(majority, agreement, true_labels, synthetic_labels, threshold=0.5):
    accepted = agreement >= threshold
    cleaned = torch.where(accepted, majority, synthetic_labels)
    return {
        "synthetic_accuracy": (synthetic_labels == true_labels).float().mean().item(),
        "cleaned_accuracy": (cleaned == true_labels).float().mean().item(),
        "auto_accept_rate": accepted.float().mean().item(),
        "triage_rate": (~accepted).float().mean().item(),
    }
