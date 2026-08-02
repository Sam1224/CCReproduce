import torch
from torch import nn
import torch.nn.functional as functional


class ToyFrozenVLM(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int, value_layers: int = 4):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.value_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(value_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        if image_dim == hidden_dim:
            nn.init.eye_(self.image_proj.weight)
            nn.init.zeros_(self.image_proj.bias)
        if text_dim == hidden_dim:
            nn.init.eye_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)
        for layer in self.value_layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, frame_features: torch.Tensor, question_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_states = torch.tanh(self.image_proj(frame_features))
        question_state = torch.tanh(self.text_proj(question_features)).unsqueeze(1)
        value_cache = []
        hidden_states = frame_states
        for layer in self.value_layers:
            contextual_states = hidden_states + 0.15 * question_state
            values = self.norm(torch.tanh(layer(contextual_states)))
            value_cache.append(values)
            hidden_states = hidden_states + 0.25 * values
        return hidden_states, value_cache[-1]


class ReTokenRetriever(nn.Module):
    def __init__(self, image_dim: int = 64, text_dim: int = 64, hidden_dim: int = 64, value_layers: int = 4):
        super().__init__()
        self.vlm = ToyFrozenVLM(image_dim=image_dim, text_dim=text_dim, hidden_dim=hidden_dim, value_layers=value_layers)
        self.retoken_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.retoken_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.retoken_projection.weight)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.answer_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def score_frames(self, frame_features: torch.Tensor, question_features: torch.Tensor) -> torch.Tensor:
        _, final_value_cache = self.vlm(frame_features, question_features)
        contextual_retoken = self.retoken_embedding.unsqueeze(0) + torch.tanh(self.vlm.text_proj(question_features))
        retrieval_query = self.retoken_projection(contextual_retoken)
        retrieval_query = functional.normalize(retrieval_query, dim=-1)
        value_vectors = functional.normalize(final_value_cache, dim=-1)
        return torch.einsum("bd,bfd->bf", retrieval_query, value_vectors)

    def retrieval_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.logit_scale.clamp(1.0, 30.0) * scores
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        positive_loss = functional.binary_cross_entropy_with_logits(logits[positive_mask], labels[positive_mask]) if positive_mask.any() else torch.tensor(0.0, device=scores.device)
        negative_loss = functional.binary_cross_entropy_with_logits(logits[negative_mask], labels[negative_mask]) if negative_mask.any() else torch.tensor(0.0, device=scores.device)
        return positive_loss + negative_loss

    def answer_logits(self, frame_features: torch.Tensor, question_features: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        scores = self.score_frames(frame_features, question_features)
        selected_indices = scores.topk(k=top_k, dim=1).indices
        gathered = torch.gather(frame_features, dim=1, index=selected_indices.unsqueeze(-1).expand(-1, -1, frame_features.size(-1)))
        pooled = gathered.mean(dim=1)
        hidden = torch.tanh(nn.functional.pad(pooled, (0, self.answer_head[0].normalized_shape[0] - pooled.size(-1))))
        return self.answer_head(hidden)

    def forward(self, frame_features: torch.Tensor, question_features: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        scores = self.score_frames(frame_features, question_features)
        outputs = {"scores": scores}
        if labels is not None:
            outputs["loss"] = self.retrieval_loss(scores, labels)
        return outputs
