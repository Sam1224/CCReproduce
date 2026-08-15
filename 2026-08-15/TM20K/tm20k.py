import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMerger(nn.Module):
    def __init__(self, strategy="uniform", merge_ratio=0.25):
        super().__init__()
        if not 0 < merge_ratio <= 1:
            raise ValueError("merge_ratio must be in (0, 1].")
        self.strategy = strategy
        self.merge_ratio = merge_ratio

    def forward(self, embeddings, importance=None):
        batch_size, seq_len, hidden = embeddings.shape
        target_len = max(1, int(math.ceil(seq_len * self.merge_ratio)))
        if target_len >= seq_len:
            return embeddings
        if self.strategy == "recent_keep":
            recent_len = target_len // 2
            pooled_len = target_len - recent_len
            old_tokens = embeddings[:, : seq_len - recent_len]
            recent_tokens = embeddings[:, seq_len - recent_len :]
            pooled = self._adaptive_average(old_tokens, pooled_len)
            return torch.cat([pooled, recent_tokens], dim=1)
        if self.strategy == "importance" and importance is not None:
            topk = torch.topk(importance, k=target_len, dim=1).indices.sort(dim=1).values
            gather_index = topk.unsqueeze(-1).expand(-1, -1, hidden)
            return embeddings.gather(1, gather_index)
        return self._adaptive_average(embeddings, target_len)

    @staticmethod
    def _adaptive_average(embeddings, target_len):
        transposed = embeddings.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(transposed, target_len)
        return pooled.transpose(1, 2)


class TM20KEncoder(nn.Module):
    def __init__(self, vocab_size=4096, num_categories=64, hidden_size=128, num_layers=2, num_heads=4, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def embed(self, tokens, categories, positions):
        return self.norm(
            self.item_embedding(tokens)
            + self.category_embedding(categories)
            + self.position_embedding(positions.clamp_max(self.position_embedding.num_embeddings - 1))
        )

    def encode_embeddings(self, embeddings, key_padding_mask=None):
        sequence = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        pooled = sequence.mean(dim=1)
        return sequence, pooled

    def forward(self, tokens, categories, positions, embedded_override=None):
        embeddings = embedded_override if embedded_override is not None else self.embed(tokens, categories, positions)
        return self.encode_embeddings(embeddings)


class TM20KRanker(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.norm.normalized_shape[0], encoder.norm.normalized_shape[0]),
            nn.GELU(),
            nn.Linear(encoder.norm.normalized_shape[0], num_classes),
        )

    def forward(self, tokens, categories, positions, embedded_override=None):
        sequence, pooled = self.encoder(tokens, categories, positions, embedded_override=embedded_override)
        return {"sequence": sequence, "pooled": pooled, "logits": self.classifier(pooled)}


class TM20KStudent(nn.Module):
    def __init__(self, base_ranker, merger):
        super().__init__()
        self.base_ranker = base_ranker
        self.merger = merger

    def forward(self, tokens, categories, positions, importance=None):
        embeddings = self.base_ranker.encoder.embed(tokens, categories, positions)
        merged = self.merger(embeddings, importance=importance)
        dummy_tokens = tokens[:, : merged.size(1)]
        dummy_categories = categories[:, : merged.size(1)]
        dummy_positions = torch.arange(merged.size(1), device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        return self.base_ranker(dummy_tokens, dummy_categories, dummy_positions, embedded_override=merged)


def tm20k_distillation_loss(student_output, teacher_output, labels, alpha=0.5, beta=0.2, temperature=2.0):
    supervised = F.cross_entropy(student_output["logits"], labels)
    soft_teacher = F.softmax(teacher_output["logits"] / temperature, dim=-1)
    soft_student = F.log_softmax(student_output["logits"] / temperature, dim=-1)
    logit_kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * temperature * temperature
    representation_kd = F.mse_loss(student_output["pooled"], teacher_output["pooled"].detach())
    return supervised + alpha * logit_kd + beta * representation_kd
