"""Interest profile tokens + Transformer-decoder generative recommender (Sec. 3.3).

Prototype embeddings  V = (P^T X) / (P^T 1)            (Eq. 6, C x d)
Interest profile token Y = P V                          (Eq. 7, |I| x d)
Sequence              R_u = [BOS, x_i1, y_i1, ...]      (Eq. 8)
The decoder-only Transformer predicts (a) the next item (L_item, Eqs. 9-10) at
post-profile positions, and (b) item interest profile (L_profile, Eqs. 11-12) at
item positions. Loss L = L_item + lambda * L_profile (Eq. 13).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def interest_prototypes(P, X_items):
    """Eq. 6: V = (P^T X) / (P^T 1)."""
    num = P.t() @ X_items                                  # [C, d]
    den = P.sum(0).clamp_min(1e-8).unsqueeze(1)            # [C, 1]
    return num / den


def profile_tokens(P, V):
    """Eq. 7: Y = P V."""
    return P @ V                                           # [|I|, d]


class G2Rec(nn.Module):
    def __init__(self, X_full, P, C, d_model=128, nhead=4, nlayers=2,
                 max_items=50, dropout=0.1):
        super().__init__()
        num_items_p1, d = X_full.shape                     # includes PAD row 0
        self.num_items = num_items_p1 - 1
        self.C = C
        self.register_buffer("X", torch.as_tensor(X_full, dtype=torch.float32))
        # P aligned to item ids 1..N with a zero PAD row at index 0
        P_full = torch.zeros(num_items_p1, C)
        P_full[1:] = P
        self.register_buffer("P_full", P_full)
        V = interest_prototypes(P, self.X[1:])             # [C, d]
        Y_full = torch.zeros(num_items_p1, d)
        Y_full[1:] = profile_tokens(P, V)                  # Eq. 7
        self.register_buffer("Y_full", Y_full)
        self.register_buffer("V", V)
        # input projections + token-type / positional embeddings
        self.in_proj = nn.Linear(d, d_model)
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)
        self.type_emb = nn.Embedding(3, d_model)           # 0=BOS,1=item,2=profile
        self.pos_emb = nn.Embedding(2 * max_items + 1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout,
                                           batch_first=True)
        self.dec = nn.TransformerEncoder(layer, nlayers)
        self.item_head = nn.Linear(d_model, self.num_items + 1)  # over item vocab
        self.interest_head = nn.Linear(d_model, C)              # over prototypes
        self.max_items = max_items

    def _build_sequence(self, seqs):
        """seqs:[B,N] item ids -> token embeddings [B,2N+1,d_model] + type ids."""
        B, N = seqs.shape
        x = self.in_proj(self.X[seqs])                     # [B,N,dm] item tokens
        y = self.in_proj(self.Y_full[seqs])               # [B,N,dm] profile tokens
        dm = x.shape[-1]
        tok = torch.zeros(B, 2 * N + 1, dm, device=x.device)
        tok[:, 0] = self.bos
        tok[:, 1::2] = x                                   # x_t at 2t+1
        tok[:, 2::2] = y                                   # y_t at 2t+2
        types = torch.empty(B, 2 * N + 1, dtype=torch.long, device=x.device)
        types[:, 0] = 0
        types[:, 1::2] = 1
        types[:, 2::2] = 2
        pos = torch.arange(2 * N + 1, device=x.device).unsqueeze(0)
        tok = tok + self.type_emb(types) + self.pos_emb(pos)
        return tok

    def forward(self, seqs, lengths):
        """Return item_logits, interest_logits and the hidden states."""
        B, N = seqs.shape
        tok = self._build_sequence(seqs)
        L = tok.shape[1]
        causal = torch.triu(torch.ones(L, L, device=tok.device), 1).bool()
        # key padding mask: token valid iff its item index < length
        item_idx = (torch.arange(N, device=tok.device).unsqueeze(0)
                    .expand(B, N))                         # [B,N]
        valid_item = item_idx < lengths.unsqueeze(1)       # [B,N]
        kpm = torch.ones(B, L, dtype=torch.bool, device=tok.device)
        kpm[:, 0] = False
        kpm[:, 1::2] = ~valid_item
        kpm[:, 2::2] = ~valid_item
        h = self.dec(tok, mask=causal, src_key_padding_mask=kpm)  # [B,L,dm]
        item_logits = self.item_head(h)
        interest_logits = self.interest_head(h)
        return item_logits, interest_logits, h

    def loss(self, seqs, lengths, lam=0.5):
        """L = L_item + lambda * L_profile (Eq. 13)."""
        B, N = seqs.shape
        item_logits, interest_logits, _ = self.forward(seqs, lengths)
        device = seqs.device
        # ---- L_item (Eqs. 9-10): hidden at y_t (idx 2t+2) -> item_{t+1} ----
        y_pos = torch.arange(N, device=device) * 2 + 2     # positions of y_t
        logits_next = item_logits[:, y_pos]                # [B,N,vocab]
        target_next = torch.zeros(B, N, dtype=torch.long, device=device)
        target_next[:, :-1] = seqs[:, 1:]                  # next item id
        valid_next = (torch.arange(N, device=device).unsqueeze(0)
                      < (lengths - 1).unsqueeze(1))        # t = 0..len-2
        li = F.cross_entropy(logits_next[valid_next], target_next[valid_next])
        # ---- L_profile (Eqs. 11-12): hidden at x_t (idx 2t+1) -> p_{i_t} ----
        x_pos = torch.arange(N, device=device) * 2 + 1
        logp = F.log_softmax(interest_logits[:, x_pos], dim=-1)  # [B,N,C]
        soft = self.P_full[seqs]                           # [B,N,C] soft labels
        valid_cur = (torch.arange(N, device=device).unsqueeze(0)
                     < lengths.unsqueeze(1))
        lp = -(soft * logp).sum(-1)[valid_cur].mean()
        return li + lam * lp, li.detach(), lp.detach()

    @torch.no_grad()
    def score_last(self, hist, lengths):
        """Next-item logits at the last valid position (after last y)."""
        item_logits, _, _ = self.forward(hist, lengths)
        last_y = (lengths * 2)                             # idx 2*len  (last y_t)
        idx = last_y.clamp(max=item_logits.shape[1] - 1)
        return item_logits[torch.arange(hist.shape[0]), idx]  # [B, vocab]
