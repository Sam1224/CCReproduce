"""
Generative Relevance Model (GRM).

From paper §3.2 / §4:
    "A generative relevance model (GRM) is trained on the labeled data from the Annotator Agent.
     GRM is a fine-tuned language model that predicts both the relevance label and a natural
     language explanation for each query-product pair."

Architecture (paper-faithful):
    Input:  [CLS] query [SEP] product_title [SEP] product_desc [SEP]
    Output: label ∈ {exact, substitute, complement, irrelevant}
            + explanation text (generative)

Training:
    Loss = α * CrossEntropy(label) + (1 - α) * NLL(explanation)

This reproduction implements a classification-only GRM using a small transformer
(DistilBERT-style) for runability without GPU. The generative explanation head
is represented as pseudocode + formula.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


LABEL_SPACE = ["exact", "substitute", "complement", "irrelevant"]
NUM_LABELS = len(LABEL_SPACE)


@dataclass
class GRMOutput:
    logits: torch.Tensor           # (batch, num_labels)
    label_probs: torch.Tensor      # (batch, num_labels) after softmax
    predicted_labels: list         # list of string labels
    loss: Optional[torch.Tensor] = None


class QueryProductEncoder(nn.Module):
    """
    Encodes (query, product_title, product_desc) into a fixed-size representation.

    Toy implementation: bag-of-words TF-IDF-like embedding via an embedding lookup
    and mean pooling. In production: replace with a pre-trained LLM encoder.
    """

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.product_proj = nn.Linear(embed_dim * 2, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _tokenize(self, texts: list, max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Toy character-level n-gram tokenization (no external tokenizer needed)."""
        all_ids, all_offsets = [], []
        offset = 0
        for text in texts:
            tokens = [hash(text[i:i+3]) % 10000 for i in range(min(len(text), max_len))]
            if not tokens:
                tokens = [0]
            all_ids.extend(tokens)
            all_offsets.append(offset)
            offset += len(tokens)
        ids = torch.tensor(all_ids, dtype=torch.long)
        offsets = torch.tensor(all_offsets, dtype=torch.long)
        return ids, offsets

    def forward(
        self,
        queries: list,
        titles: list,
        descs: list,
    ) -> torch.Tensor:
        q_ids, q_off = self._tokenize(queries)
        q_emb = self.embedding(q_ids, q_off)           # (batch, embed_dim)
        q_enc = F.gelu(self.query_proj(q_emb))

        t_ids, t_off = self._tokenize(titles)
        t_emb = self.embedding(t_ids, t_off)
        d_ids, d_off = self._tokenize(descs)
        d_emb = self.embedding(d_ids, d_off)
        p_cat = torch.cat([t_emb, d_emb], dim=-1)      # (batch, embed_dim*2)
        p_enc = F.gelu(self.product_proj(p_cat))

        fused = torch.cat([q_enc, p_enc], dim=-1)       # (batch, hidden_dim*2)
        out = self.norm(F.gelu(self.fusion(fused)))      # (batch, hidden_dim)
        return out


class GenerativeRelevanceModel(nn.Module):
    """
    GRM: Generative Relevance Model.

    Classification head (implemented):
        logits = W_cls * encoder(query, title, desc)  ∈ R^{num_labels}
        L_cls = CrossEntropy(logits, gold_label)

    Generative explanation head (pseudocode only — requires autoregressive decoder):
        # decoder = AutoregressiveDecoder(hidden_dim, vocab_size)
        # explanation_tokens = decoder.generate(encoder_hidden, max_len=64)
        # L_gen = NLL(explanation_tokens, gold_explanation)
        # Total loss = alpha * L_cls + (1 - alpha) * L_gen

    Full training objective (paper §3.2, eq. not specified, inferred):
        L = α * L_cls + (1 - α) * L_gen
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.alpha = alpha
        self.encoder = QueryProductEncoder(vocab_size, embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )
        self.label_space = LABEL_SPACE

    def forward(
        self,
        queries: list,
        titles: list,
        descs: list,
        labels: Optional[torch.Tensor] = None,
    ) -> GRMOutput:
        hidden = self.encoder(queries, titles, descs)   # (B, hidden_dim)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)                # (B, num_labels)
        probs = F.softmax(logits, dim=-1)
        predicted = [self.label_space[i] for i in logits.argmax(dim=-1).tolist()]

        loss = None
        if labels is not None:
            # L_cls = CrossEntropy(logits, labels)
            loss_cls = F.cross_entropy(logits, labels)
            # L_gen: pseudocode only
            # loss_gen = self._generative_loss(hidden, gold_explanations)
            # loss = self.alpha * loss_cls + (1 - self.alpha) * loss_gen
            loss = loss_cls  # alpha * L_cls (L_gen omitted in toy)

        return GRMOutput(
            logits=logits,
            label_probs=probs,
            predicted_labels=predicted,
            loss=loss,
        )

    def predict(self, query: str, title: str, desc: str) -> Tuple[str, float]:
        self.eval()
        with torch.no_grad():
            out = self.forward([query], [title], [desc])
        label = out.predicted_labels[0]
        conf = out.label_probs[0].max().item()
        return label, conf
