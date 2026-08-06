import torch
from torch import nn
import torch.nn.functional as functional


class MMoEReward(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_experts)])
        self.click_gate = nn.Linear(feature_dim, num_experts)
        self.purchase_gate = nn.Linear(feature_dim, num_experts)
        self.click_head = nn.Linear(hidden_dim, 1)
        self.purchase_head = nn.Linear(hidden_dim, 1)
        self.explore_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _mix(self, item_features, gate):
        expert_stack = torch.stack([expert(item_features) for expert in self.experts], dim=-2)
        weights = torch.softmax(gate(item_features), dim=-1).unsqueeze(-1)
        return (expert_stack * weights).sum(dim=-2)

    def forward(self, item_features):
        click_state = self._mix(item_features, self.click_gate)
        purchase_state = self._mix(item_features, self.purchase_gate)
        click_score = self.click_head(click_state).squeeze(-1)
        purchase_score = self.purchase_head(purchase_state).squeeze(-1)
        sequence_state = click_state.mean(dim=1)
        explore_score = self.explore_head(sequence_state).squeeze(-1)
        return click_score, purchase_score, explore_score


class DEGRGenerator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, slate_size=10, num_heads=4, exploration_alpha=0.2):
        super().__init__()
        self.slate_size = slate_size
        self.exploration_alpha = exploration_alpha
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, batch_first=True), num_layers=1)
        self.input_projection = nn.Linear(feature_dim + 3, hidden_dim)
        self.pointer = nn.Linear(hidden_dim, 1)

    def forward(self, item_features, click_score, purchase_score, explore_score, target_order=None):
        request_count, candidate_count, _ = item_features.shape
        explore_channel = explore_score[:, None, None].expand(request_count, candidate_count, 1)
        model_input = torch.cat([item_features, click_score.unsqueeze(-1), purchase_score.unsqueeze(-1), explore_channel], dim=-1)
        encoded = self.encoder(self.input_projection(model_input))
        pointer_logits = self.pointer(encoded).squeeze(-1)
        blended_logits = pointer_logits + self.exploration_alpha * explore_score[:, None]
        if target_order is None:
            return self.greedy_decode(blended_logits)
        log_probs = functional.log_softmax(blended_logits, dim=-1)
        supervised_loss = -torch.gather(log_probs, 1, target_order).mean()
        diversity_loss = self._diversity_penalty(encoded, target_order)
        return blended_logits, supervised_loss, diversity_loss

    def greedy_decode(self, logits):
        return torch.topk(logits, k=self.slate_size, dim=-1).indices

    def _diversity_penalty(self, encoded_items, selected_indices):
        selected = torch.gather(encoded_items, 1, selected_indices.unsqueeze(-1).expand(-1, -1, encoded_items.size(-1)))
        normalized = functional.normalize(selected, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        mask = 1.0 - torch.eye(selected_indices.size(1), device=selected_indices.device).unsqueeze(0)
        return (similarity * mask).pow(2).mean()


def adaptive_orpo_loss(policy_logits, target_order, reward, temperature=0.2):
    chosen_logprob = torch.gather(functional.log_softmax(policy_logits, dim=-1), 1, target_order).mean(dim=1)
    rejected_indices = torch.argsort(policy_logits, dim=1)[:, : target_order.size(1)]
    rejected_logprob = torch.gather(functional.log_softmax(policy_logits, dim=-1), 1, rejected_indices).mean(dim=1)
    reward_weight = torch.sigmoid(reward.detach())
    return -functional.logsigmoid((chosen_logprob - rejected_logprob) / temperature).mul(reward_weight).mean()
