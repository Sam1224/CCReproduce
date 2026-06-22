"""
G2Rec — Structuring and Tokenizing Distributed User Interest Context
for Generative Recommendation (arXiv 2606.20554)

Architecture overview:
  1. HolisticCoEngagementGraph: scalable graph that captures GLOBAL user
     co-engagement patterns to derive user-interest prototypes (no GNN —
     uses spectral / random-walk style propagation that avoids O(n²) GNN).
  2. SupervisedTokenizer (RQ-VAE style): residual quantization to tokenize
     items into discrete code sequences, supervised by the interest-prototype
     soft-labels from step 1.
  3. AutoregressiveRecommender (Transformer decoder): generates the next
     item token sequence autoregressively, conditioned on the user's past
     interaction token sequences.

Paper formula references:
  - Eq (interest prototype assignment): P(z_u | u) = softmax(W·h_u)   [§3.2]
  - Eq (tokenization loss): L_tok = CE(logits_tok, z_proto_soft_label) [§3.3]
  - Eq (recommendation loss): L_rec = CE(logits_next, target_codes)    [§3.4]
  - Total loss: L = L_rec + λ · L_tok                                  [§3.5]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Holistic Co-Engagement Graph — interest prototype derivation
# ---------------------------------------------------------------------------

class HolisticCoEngagementGraph(nn.Module):
    """
    Derives interest-prototype soft assignments for items from the
    global co-engagement graph WITHOUT running a GNN.

    We use a scalable random-walk-based propagation:
      H_items = D^{-1/2} A D^{-1/2} E_items   (one-hop smoothing)
    where A is the item-item co-occurrence matrix (users are marginalized out),
    and E_items is the learnable item embedding table.

    The smoothed embeddings are projected to K interest prototypes.

    In the paper (§3.2) the co-engagement graph is built from user-item
    interactions; we approximate with normalized item-item co-occurrence.
    """

    def __init__(self, n_items: int, embed_dim: int, n_prototypes: int):
        super().__init__()
        self.n_items = n_items
        self.n_prototypes = n_prototypes
        self.embed_dim = embed_dim

        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        # Projection to prototype logits
        self.proto_proj = nn.Linear(embed_dim, n_prototypes)

    def build_cooccurrence(self, user_seqs: torch.Tensor) -> torch.Tensor:
        """
        Build a row-normalized item-item co-occurrence matrix from user sequences.
        user_seqs : (B, L) padded item IDs (0 = pad)
        Returns   : sparse adjacency (n_items+1, n_items+1) on CPU for precomputation
        """
        # Pseudocode:
        # For each user sequence, all item pairs (i, j) that co-occur get
        # their co-occurrence count incremented.  Then row-normalize.
        # In production this is precomputed offline.  Here we return identity
        # as a placeholder so the model can still train.
        n = self.n_items + 1
        # Identity propagation (no smoothing) — replace with real co-occurrence
        # matrix in production.
        return torch.eye(n, device=user_seqs.device)

    def forward(self, item_ids: torch.Tensor, user_seqs: torch.Tensor = None):
        """
        item_ids  : (N,) item IDs
        user_seqs : (B, L) — used to build co-occurrence (optional at inference)
        Returns
          proto_logits : (N, K) soft prototype assignment logits
        """
        # Raw item embeddings
        E = self.item_emb(item_ids)  # (N, D)

        # Graph smoothing — in paper this uses the precomputed co-engagement
        # propagation; here we do a no-op (identity) for the toy version.
        # Real impl: multiply by normalized co-occurrence matrix row.
        E_smooth = E  # (N, D) — placeholder; replace with graph propagation

        proto_logits = self.proto_proj(E_smooth)  # (N, K)
        return proto_logits


# ---------------------------------------------------------------------------
# 2. Supervised Tokenizer — RQ-VAE with prototype supervision
# ---------------------------------------------------------------------------

class ResidualQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RQ-VQ): encodes an embedding as a sequence
    of M discrete codes, each chosen from a codebook of size V.

    Ref: SoundStream / TIGER for the basic RQ structure.
    G2Rec modification: adds a prototype-supervised loss on the first code.
    """

    def __init__(self, embed_dim: int, codebook_size: int, n_codes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.n_codes = n_codes  # M — number of residual steps

        # One codebook per residual step
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, embed_dim))
            for _ in range(n_codes)
        ])

    def forward(self, z: torch.Tensor):
        """
        z : (N, D) input embeddings
        Returns
          codes     : (N, M) discrete code indices
          z_q       : (N, D) quantized embedding (straight-through)
          commit_loss : scalar commitment loss
        """
        residual = z
        codes = []
        z_q_accum = torch.zeros_like(z)
        commit_loss = 0.0

        for cb in self.codebooks:
            # Nearest-centroid lookup in this codebook
            # distances: (N, V)
            dist = (
                residual.pow(2).sum(-1, keepdim=True)
                - 2 * residual @ cb.T
                + cb.pow(2).sum(-1)
            )
            code = dist.argmin(dim=-1)  # (N,)
            codes.append(code)

            # Quantized vector (straight-through gradient)
            z_hat = cb[code]  # (N, D)
            commit_loss = commit_loss + F.mse_loss(z_hat.detach(), residual)
            z_q_accum = z_q_accum + residual + (z_hat - residual).detach()
            residual = residual - z_hat.detach()

        codes = torch.stack(codes, dim=1)  # (N, M)
        return codes, z_q_accum, commit_loss


class SupervisedTokenizer(nn.Module):
    """
    Tokenizer with prototype supervision (G2Rec §3.3).

    Encodes items into discrete code sequences via RQ-VAE.
    A cross-entropy loss aligns the first code with the proto-softlabel
    derived from the HolisticCoEngagementGraph.

    Loss: L_tok = CE(first_code_logit, argmax(proto_logit)) + β * commit_loss
    """

    def __init__(
        self,
        n_items: int,
        embed_dim: int,
        codebook_size: int,
        n_codes: int,
        n_prototypes: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(n_items + 1, embed_dim, padding_idx=0),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.rq = ResidualQuantizer(embed_dim, codebook_size, n_codes)
        # Map first-code logits → prototype space for supervision
        self.proto_head = nn.Linear(codebook_size, n_prototypes)

    def encode(self, item_ids: torch.Tensor):
        """item_ids : (N,) → codes : (N, M), z_q : (N, D)"""
        # Get embeddings from just the Embedding layer
        emb = self.encoder[0](item_ids)  # (N, D)
        # Apply remaining layers
        z = self.encoder[1](emb)
        z = self.encoder[2](z)
        z = self.encoder[3](z)
        codes, z_q, commit_loss = self.rq(z)
        return codes, z_q, commit_loss

    def tokenization_loss(
        self, item_ids: torch.Tensor, proto_logits: torch.Tensor
    ):
        """
        Compute supervised tokenization loss.

        proto_logits : (N, K) soft prototype assignments from graph module
        Returns scalar loss.
        """
        codes, z_q, commit_loss = self.encode(item_ids)

        # Pseudo-label: argmax of prototype logits (hard assignment)
        proto_labels = proto_logits.argmax(dim=-1)  # (N,)

        # First residual code's "soft" logit — project first codebook distances
        # to prototype space.  (Simplified: we use z_q projected to prototypes.)
        # In the paper a dedicated projection is trained; here we reuse proto_head.
        logit_for_proto = self.proto_head(z_q @ self.rq.codebooks[0].T)  # (N, K)

        tok_loss = F.cross_entropy(logit_for_proto, proto_labels)
        return tok_loss + 0.1 * commit_loss


# ---------------------------------------------------------------------------
# 3. Autoregressive Recommender — Transformer Decoder
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1)]


class AutoregressiveRecommender(nn.Module):
    """
    Causal Transformer that generates next-item codes autoregressively.

    Input : sequence of flattened item codes (each item → M tokens)
    Output: logits over codebook at each position
    """

    def __init__(
        self,
        vocab_size: int,       # = codebook_size (shared across all RQ levels)
        n_codes: int,          # M residual codes per item
        seq_len: int,          # max number of items in history
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codes = n_codes
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len * n_codes + n_codes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, code_seq: torch.Tensor):
        """
        code_seq : (B, T) flattened code tokens (0-padded)
        Returns logits (B, T, V)
        """
        x = self.token_emb(code_seq)   # (B, T, D)
        x = self.pos_enc(x)
        T = x.size(1)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.head(x)          # (B, T, V)
        return logits


# ---------------------------------------------------------------------------
# 4. G2Rec — unified model
# ---------------------------------------------------------------------------

class G2Rec(nn.Module):
    """
    Full G2Rec model.

    Forward pass (training):
      1. HolisticCoEngagementGraph produces proto_logits for all items in batch
      2. SupervisedTokenizer encodes history + target items into code sequences,
         computing tokenization_loss using proto_logits as supervision
      3. AutoregressiveRecommender predicts next item's codes given history codes
      Total loss = L_rec + λ * L_tok   (Eq. from paper §3.5)
    """

    def __init__(
        self,
        n_items: int,
        embed_dim: int = 64,
        codebook_size: int = 256,
        n_codes: int = 4,
        n_prototypes: int = 32,
        seq_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        lam: float = 0.5,  # λ: weight of tokenization loss
    ):
        super().__init__()
        self.n_items = n_items
        self.n_codes = n_codes
        self.lam = lam

        self.graph = HolisticCoEngagementGraph(n_items, embed_dim, n_prototypes)
        self.tokenizer = SupervisedTokenizer(
            n_items, embed_dim, codebook_size, n_codes, n_prototypes
        )
        self.recommender = AutoregressiveRecommender(
            vocab_size=codebook_size,
            n_codes=n_codes,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

    def _encode_seq(self, item_seqs: torch.Tensor):
        """
        item_seqs : (B, L) padded item IDs
        Returns   : (B, L*M) flat code sequence
        """
        B, L = item_seqs.shape
        flat_items = item_seqs.reshape(-1)  # (B*L,)
        codes, _, _ = self.tokenizer.encode(flat_items)  # (B*L, M)
        # Zero out padding positions
        pad_mask = (flat_items == 0).unsqueeze(-1).expand_as(codes)
        codes = codes.masked_fill(pad_mask, 0)
        return codes.reshape(B, L * self.n_codes)  # (B, L*M)

    def forward(self, user_seqs: torch.Tensor, targets: torch.Tensor):
        """
        user_seqs : (B, L) padded item ID sequences
        targets   : (B,) next-item IDs

        Returns (total_loss, rec_loss, tok_loss)
        """
        B, L = user_seqs.shape

        # --- Step 1: proto logits for all unique items ---
        all_items = torch.cat([user_seqs.reshape(-1), targets]).unique()
        all_items = all_items[all_items > 0]
        proto_logits = self.graph(all_items, user_seqs)  # (N_unique, K)

        # --- Step 2: tokenization loss ---
        tok_loss = self.tokenizer.tokenization_loss(all_items, proto_logits)

        # --- Step 3: encode history + target ---
        hist_codes = self._encode_seq(user_seqs)          # (B, L*M)
        tgt_codes, _, _ = self.tokenizer.encode(targets)  # (B, M)

        # Teacher-forced: predict target codes given history codes
        # Input sequence: [hist_codes | tgt_codes[:-1]]
        dec_in = torch.cat([hist_codes, tgt_codes[:, :-1]], dim=1)  # (B, L*M + M-1)
        logits = self.recommender(dec_in)  # (B, L*M + M-1, V)

        # Reconstruction loss only over the target positions
        # target positions: last M-1 + the last one → tgt_codes
        rec_logits = logits[:, -self.n_codes :, :]  # (B, M, V)
        rec_loss = F.cross_entropy(
            rec_logits.reshape(B * self.n_codes, -1),
            tgt_codes.reshape(B * self.n_codes),
            ignore_index=0,
        )

        total_loss = rec_loss + self.lam * tok_loss
        return total_loss, rec_loss, tok_loss

    @torch.no_grad()
    def generate(self, user_seqs: torch.Tensor, top_k: int = 10):
        """
        Beam / greedy generation of the next item's codes, then
        retrieve nearest item from codebook.

        user_seqs : (B, L) padded item IDs
        Returns   : (B, top_k) predicted item IDs (approximate)
        """
        B, L = user_seqs.shape
        hist_codes = self._encode_seq(user_seqs)  # (B, L*M)

        pred_codes = []
        inp = hist_codes  # (B, L*M)
        for step in range(self.n_codes):
            logits = self.recommender(inp)       # (B, T, V)
            next_code = logits[:, -1, :].argmax(-1)  # (B,) greedy
            pred_codes.append(next_code)
            inp = torch.cat([inp, next_code.unsqueeze(-1)], dim=1)

        pred_codes = torch.stack(pred_codes, dim=1)  # (B, M)

        # Nearest-item retrieval: for each item, compute its code sequence;
        # return items whose codes best match pred_codes.
        # (Simplified: return random top-k as placeholder — real impl uses
        # an inverted index over the item code table.)
        all_item_ids = torch.arange(1, self.n_items + 1, device=user_seqs.device)
        return all_item_ids[:top_k].unsqueeze(0).expand(B, -1)
