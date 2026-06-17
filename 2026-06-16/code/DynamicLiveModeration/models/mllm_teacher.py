"""
MLLM Teacher model for knowledge distillation.

Paper (§3.3): "an MLLM distilling knowledge into each [pipeline] to boost
accuracy while keeping inference lightweight."

In the full system, the teacher is a large MLLM (e.g., GPT-4V scale).
Here we approximate it with a larger transformer + soft labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """Fuse text, audio, and visual modalities."""

    def __init__(self, text_dim=128, audio_dim=512, visual_dim=768, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, audio_feat, visual_feat):
        # Project each modality to common hidden dim
        t = self.text_proj(text_feat).unsqueeze(1)       # (B, 1, H)
        a = self.audio_proj(audio_feat).unsqueeze(1)     # (B, 1, H)
        v = self.visual_proj(visual_feat).unsqueeze(1)   # (B, 1, H)

        # Stack modalities as sequence
        seq = torch.cat([t, a, v], dim=1)  # (B, 3, H)

        # Cross-modal attention
        fused, _ = self.cross_attn(seq, seq, seq)
        fused = self.norm(seq + fused)

        # Mean pooling
        return fused.mean(dim=1)  # (B, H)


class MLLMTeacher(nn.Module):
    """
    Large MLLM teacher model.

    Paper: The MLLM generates soft labels (logits) used to distill knowledge
    into the lighter classification and similarity pipelines.

    Architecture: Deep multimodal transformer with large capacity.
    """

    def __init__(self, text_dim=128, audio_dim=512, visual_dim=768,
                 hidden_dim=1024, num_classes=2, num_layers=6):
        super().__init__()
        self.fusion = MultimodalFusion(text_dim, audio_dim, visual_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=16, dim_feedforward=4096,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Embedding head for similarity pipeline distillation
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 512),  # 512-d embedding
        )

    def forward(self, text_feat, audio_feat, visual_feat):
        """
        Returns:
            logits: (B, num_classes)
            embeddings: (B, 512) — used for similarity matching
        """
        fused = self.fusion(text_feat, audio_feat, visual_feat)
        # Transformer processes the fused representation
        encoded = self.transformer(fused.unsqueeze(1)).squeeze(1)

        logits = self.classifier(encoded)
        embeddings = F.normalize(self.embedding_head(encoded), dim=-1)
        return logits, embeddings

    @torch.no_grad()
    def get_soft_labels(self, text_feat, audio_feat, visual_feat, temperature=3.0):
        """Generate soft labels for knowledge distillation (temperature scaling)."""
        logits, embeddings = self(text_feat, audio_feat, visual_feat)
        soft_labels = F.softmax(logits / temperature, dim=-1)
        return soft_labels, embeddings


class TextEncoder(nn.Module):
    """Simple text encoder to produce text_feat from token sequences."""

    def __init__(self, vocab_size=1000, embed_dim=64, out_dim=128, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=4, dim_feedforward=256,
                batch_first=True, dropout=0.1,
            ),
            num_layers=2,
        )
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, token_ids):
        """token_ids: (B, L) long tensor"""
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embed(token_ids) + self.pos_embed(pos)
        x = self.encoder(x)
        return self.proj(x.mean(dim=1))  # (B, out_dim)


def simple_tokenize(texts, vocab_size=1000, max_len=64):
    """Minimal tokenizer: hash each word to an int."""
    batch = []
    for text in texts:
        tokens = [hash(w) % (vocab_size - 1) + 1 for w in text.lower().split()]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        batch.append(tokens)
    return torch.tensor(batch, dtype=torch.long)
