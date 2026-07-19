import torch
from torch import nn
import torch.nn.functional as F

from data import HEURISTICS, MANIPULATIVE_TERMS, bow, rewrite_description


class GEOReranker(nn.Module):
    def __init__(self, dim=128, hidden=64):
        super().__init__()
        self.query_proj = nn.Linear(dim, hidden)
        self.item_proj = nn.Linear(dim, hidden)
        self.scorer = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, query_features, listing_features):
        q = self.query_proj(query_features)
        x = self.item_proj(listing_features)
        q_expand = q.unsqueeze(1).expand_as(x)
        pair = torch.cat([q_expand, x, q_expand * x], dim=-1)
        return self.scorer(pair).squeeze(-1)


class PromptMetaOptimizer(nn.Module):
    def __init__(self, num_heuristics=None):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_heuristics or len(HEURISTICS)))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def choose(self):
        return HEURISTICS[int(self.weights().argmax())]


def listwise_loss(scores, labels):
    return -(torch.softmax(labels, dim=-1) * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()


def rank_improvement(ranker, query_text, original_description, target_rank=5):
    weights = []
    rewritten_scores = []
    q = bow(query_text).unsqueeze(0)
    for heuristic in HEURISTICS:
        rewritten = rewrite_description(original_description, heuristic)
        item = bow(rewritten).view(1, 1, -1)
        score = ranker(q, item).squeeze()
        penalty = sum(term in rewritten.lower() for term in MANIPULATIVE_TERMS) * 0.2
        rewritten_scores.append(score - penalty)
        weights.append(heuristic)
    stacked = torch.stack(rewritten_scores)
    return weights[int(stacked.argmax())], stacked
