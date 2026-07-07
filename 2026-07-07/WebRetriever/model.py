from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class WebRetrieverAgent(nn.Module):
    """Toy agent with retrieval, action planning, and completion heads."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64, action_steps: int = 5, action_vocab: int = 6):
        super().__init__()
        self.action_steps = action_steps
        self.action_vocab = action_vocab
        self.query_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.page_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_steps * action_vocab),
        )
        self.completion_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, pages: torch.Tensor) -> Dict[str, torch.Tensor]:
        query_state = self.query_encoder(query)
        page_states = self.page_encoder(pages)
        retrieval_logits = torch.einsum("bd,bpd->bp", query_state, page_states) / query_state.shape[-1] ** 0.5
        page_probs = retrieval_logits.softmax(dim=-1)
        retrieved_state = torch.einsum("bp,bpd->bd", page_probs, page_states)
        planner_state = torch.cat([query_state, retrieved_state], dim=-1)
        action_logits = self.action_head(planner_state).view(query.shape[0], self.action_steps, self.action_vocab)
        max_page_state = page_states.gather(
            1,
            retrieval_logits.argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, page_states.shape[-1]),
        ).squeeze(1)
        completion_logits = self.completion_head(torch.cat([query_state, retrieved_state, max_page_state], dim=-1)).squeeze(-1)
        return {
            "retrieval_logits": retrieval_logits,
            "action_logits": action_logits,
            "completion_logits": completion_logits,
            "page_probs": page_probs,
        }


def webretriever_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    retrieval = F.cross_entropy(outputs["retrieval_logits"], batch["target_page"])
    action = F.cross_entropy(outputs["action_logits"].flatten(0, 1), batch["actions"].flatten(0, 1))
    completion = F.binary_cross_entropy_with_logits(outputs["completion_logits"], batch["completed"])
    total = retrieval + 0.7 * action + 0.5 * completion
    return {
        "total": total,
        "retrieval": retrieval.detach(),
        "action": action.detach(),
        "completion": completion.detach(),
    }


@torch.no_grad()
def evaluate_outputs(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    page_arrival = (outputs["retrieval_logits"].argmax(dim=-1) == batch["target_page"]).float().mean().item()
    predicted_actions = outputs["action_logits"].argmax(dim=-1)
    action_match = (predicted_actions == batch["actions"]).float().mean().item()
    completed_pred = (outputs["completion_logits"].sigmoid() >= 0.5).float()
    completion_accuracy = (completed_pred == batch["completed"]).float().mean().item()
    return {
        "page_arrival": page_arrival,
        "action_match": action_match,
        "completion_accuracy": completion_accuracy,
    }
