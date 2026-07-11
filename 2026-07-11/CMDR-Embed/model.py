import torch
import torch.nn as nn
import torch.nn.functional as F


class CMDRChunkEncoder(nn.Module):
    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feature_dim))
        self.page_encoder = nn.Linear(feature_dim * 2, hidden_dim)
        self.contextualizer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=hidden_dim * 2),
            num_layers=1,
        )
        self.proj = nn.Linear(hidden_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, query: torch.Tensor, page_text: torch.Tensor, page_images: torch.Tensor) -> dict:
        batch_size, pages = page_text.shape[:2]
        image_features = self.image_encoder(page_images.view(batch_size * pages, 3, 16, 16)).view(batch_size, pages, -1)
        page_inputs = torch.cat([page_text, image_features], dim=-1)
        hidden = self.page_encoder(page_inputs)
        contextual = self.contextualizer(hidden)
        page_embeddings = F.normalize(self.proj(contextual), dim=-1)
        query_embedding = F.normalize(self.query_proj(query), dim=-1)
        scores = torch.einsum("bd,bpd->bp", query_embedding, page_embeddings)
        return {"query_embedding": query_embedding, "page_embeddings": page_embeddings, "scores": scores}


def cmdr_loss(outputs: dict, target: torch.Tensor, cmcl_weight: float = 0.5) -> dict:
    retrieval = F.cross_entropy(outputs["scores"], target)
    page_embeddings = outputs["page_embeddings"]
    batch_size, pages, dim = page_embeddings.shape
    same_doc_sim = torch.matmul(page_embeddings, page_embeddings.transpose(1, 2))
    eye = torch.eye(pages, device=page_embeddings.device).unsqueeze(0)
    hard_negative_penalty = (same_doc_sim * (1 - eye)).logsumexp(dim=(1, 2)).mean() / dim
    total = retrieval + cmcl_weight * hard_negative_penalty
    return {"total": total, "retrieval": retrieval.detach(), "cmcl": hard_negative_penalty.detach()}
