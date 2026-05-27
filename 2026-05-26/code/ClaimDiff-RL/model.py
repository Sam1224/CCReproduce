"""
Caption generation policy model for ClaimDiff-RL.

In the full paper, the policy is a multimodal LLM (e.g., LLaVA / InternVL variant)
that takes (image, prompt) and generates a caption.

This toy uses a small GPT-2 style language model conditioned on a text "image description"
as a stand-in.  The interface is kept identical to the real system so the training loop
(train.py) remains applicable to a real MLLM backend.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ToyCaptionPolicy(nn.Module):
    """
    Minimal autoregressive LM policy.
    Vocab is the 200 most common English words + special tokens.
    Conditioned on a bag-of-words "image feature" via a learned projection.
    """

    VOCAB_SIZE = 256       # toy vocab
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 3
    MAX_LEN = 32

    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(self.VOCAB_SIZE, self.D_MODEL, padding_idx=self.PAD_ID)
        self.pos_embed = nn.Embedding(self.MAX_LEN + 1, self.D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=self.N_HEADS,
            dim_feedforward=self.D_MODEL * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        # Image conditioning: project image_feat (D_MODEL) to a context prefix token
        self.img_proj = nn.Linear(self.D_MODEL, self.D_MODEL)
        self.lm_head = nn.Linear(self.D_MODEL, self.VOCAB_SIZE, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_image(self, image_feat: torch.Tensor) -> torch.Tensor:
        """
        image_feat: (B, D_MODEL) — toy image embedding (e.g., averaged word embeddings
        of the image description tokens).
        Returns (B, 1, D_MODEL) context prefix.
        """
        return self.img_proj(image_feat).unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,   # (B, T)
        image_feat: torch.Tensor,  # (B, D_MODEL)
    ) -> torch.Tensor:
        """Returns logits (B, T+1, VOCAB_SIZE) — including the image prefix position."""
        B, T = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings for the input sequence
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embed(input_ids) + self.pos_embed(positions)  # (B, T, D)

        # Prepend image context as position-0 token
        img_ctx = self.encode_image(image_feat)  # (B, 1, D)
        seq = torch.cat([img_ctx, tok_emb], dim=1)  # (B, T+1, D)

        # Causal mask
        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        out = self.transformer(seq, mask=causal_mask)  # (B, T+1, D)
        logits = self.lm_head(out)  # (B, T+1, VOCAB_SIZE)
        return logits

    @torch.no_grad()
    def generate(
        self,
        image_feat: torch.Tensor,  # (B, D_MODEL)
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy / temperature sampling autoregressively. Returns (B, T) token ids."""
        B = image_feat.size(0)
        device = image_feat.device

        # Start with BOS
        generated = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, image_feat)  # (B, T+1, V)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, V)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)  # (B, 1)
            generated = torch.cat([generated, next_id], dim=1)

            # Stop if all sequences have emitted EOS
            if (next_id == self.EOS_ID).all():
                break

        return generated[:, 1:]  # strip BOS

    def log_prob_of(
        self,
        input_ids: torch.Tensor,   # (B, T) — full sequence including generated tokens
        image_feat: torch.Tensor,  # (B, D_MODEL)
    ) -> torch.Tensor:
        """
        Returns per-token log probabilities (B, T-1) for the generated sequence
        (teacher-forced, used for policy gradient).
        """
        logits = self.forward(input_ids[:, :-1], image_feat)  # (B, T, V)
        # logits[:, 0, :] corresponds to predicting input_ids[:, 1]
        log_probs = F.log_softmax(logits[:, 1:, :], dim=-1)  # (B, T-1, V)
        target = input_ids[:, 1:]  # (B, T-1)
        return log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)


# ---------------------------------------------------------------------------
# Simple vocabulary helpers (toy 256-word vocab built from ASCII chars)
# ---------------------------------------------------------------------------

def build_toy_vocab():
    """Build a tiny char-level vocab as stand-in. In production: use tokenizer."""
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    for i in range(26):
        vocab[chr(ord("a") + i)] = i + 3
    for i in range(10):
        vocab[str(i)] = i + 29
    # Fill remaining with common words
    common = [
        "a", "the", "is", "in", "on", "of", "with", "and", "red", "blue", "green",
        "car", "dog", "cat", "man", "woman", "child", "park", "street", "ball",
        "playing", "sitting", "standing", "holding", "running", "white", "black",
        "brown", "wooden", "chair", "table", "flower", "tree", "house", "pizza",
        "plate", "yellow", "pink", "sunny", "near", "next", "two", "three", "four",
        "large", "small", "big", "wearing", "dress", "shirt", "hat", "basket",
        "window", "door", "sky", "grass", "water", "mountain", "city", "building",
    ]
    idx = 40
    for w in common:
        if w not in vocab:
            vocab[w] = idx
            idx += 1
            if idx >= 256:
                break
    return vocab


TOY_VOCAB = build_toy_vocab()
TOY_VOCAB_INV = {v: k for k, v in TOY_VOCAB.items()}


def encode(text: str, max_len: int = 30) -> list:
    tokens = [TOY_VOCAB.get(w, 3) for w in text.lower().split()[:max_len]]
    return [1] + tokens + [2]  # BOS + tokens + EOS


def encode_text(text: str) -> list:
    tokens = [TOY_VOCAB.get(w, 3) for w in text.lower().split()[:28]]
    return [1] + tokens + [2]


def decode(ids: list) -> str:
    return " ".join(TOY_VOCAB_INV.get(i, "?") for i in ids if i not in (0, 1, 2))


def text_to_image_feat(
    image_desc: str, d_model: int = 128, device: str = "cpu"
) -> torch.Tensor:
    """
    Convert image description to a D_MODEL-dim feature by averaging token embeddings.
    Toy approximation of visual encoder output.
    """
    emb = nn.Embedding(256, d_model)
    ids = torch.tensor([TOY_VOCAB.get(w, 3) for w in image_desc.lower().split()[:16]])
    if len(ids) == 0:
        return torch.zeros(1, d_model)
    return emb(ids).mean(0, keepdim=True).detach()
