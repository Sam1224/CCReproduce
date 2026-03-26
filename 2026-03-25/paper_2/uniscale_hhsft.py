import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper: UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking
Summary: Co-design data scaling (ES^3) and model scaling (HHSFT) for industrial e-commerce ranking.
Core: Heterogeneous Hierarchical Sample Fusion Transformer with hierarchical feature interaction and
      entire-space user interest fusion.

Note:
This is a minimal PyTorch reproduction of the *model-side core idea* (hierarchical fusion + interest pooling).
"""


class HeteroFeatureEncoder(nn.Module):
    def __init__(self, num_sparse_features=64, d_model=256):
        super().__init__()
        self.sparse_emb = nn.Embedding(num_sparse_features, d_model)
        self.dense_proj = nn.Linear(d_model, d_model)

    def forward(self, sparse_ids, dense_vec):
        sparse = self.sparse_emb(sparse_ids)
        dense = self.dense_proj(dense_vec).unsqueeze(1)
        return torch.cat([dense, sparse], dim=1)


class HierarchicalInteraction(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True)
        self.cross = nn.MultiheadAttention(d_model, nhead=nhead, batch_first=True)

    def forward(self, x, context):
        x = self.self_attn(x)
        out, _ = self.cross(query=x, key=context, value=context)
        return out


class EntireSpaceInterestFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, history_tokens):
        h, _ = self.gru(history_tokens)
        w = self.gate(h)
        return (h * w).mean(dim=1)


class HHSFT(nn.Module):
    def __init__(self, num_sparse_features=64, d_model=256):
        super().__init__()
        self.encoder = HeteroFeatureEncoder(num_sparse_features=num_sparse_features, d_model=d_model)
        self.hier = HierarchicalInteraction(d_model=d_model)
        self.interest = EntireSpaceInterestFusion(d_model=d_model)
        self.rank_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, sparse_ids, dense_vec, user_history):
        x = self.encoder(sparse_ids=sparse_ids, dense_vec=dense_vec)
        hist = self.interest(user_history)
        context = hist.unsqueeze(1).repeat(1, x.size(1), 1)
        fused = self.hier(x, context)

        pooled = fused.mean(dim=1)
        features = torch.cat([pooled, hist], dim=-1)
        return self.rank_head(features).squeeze(-1)


def ranking_loss(score, label):
    return F.binary_cross_entropy_with_logits(score, label.float())


if __name__ == "__main__":
    torch.manual_seed(0)

    bsz, seq, d = 2, 8, 256
    sparse = torch.randint(0, 64, (bsz, seq))
    dense = torch.randn(bsz, d)
    hist = torch.randn(bsz, 20, d)
    label = torch.randint(0, 2, (bsz,)).float()

    model = HHSFT(num_sparse_features=64, d_model=d)
    score = model(sparse, dense, hist)
    loss = ranking_loss(score, label)

    print("score:", score.detach().cpu().tolist())
    print("loss:", float(loss))
    print("UniScale (HHSFT) reproduction structure complete.")
