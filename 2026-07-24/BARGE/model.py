from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Catalog


class ItemContextAwareAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        expanded = self.context_proj(context).unsqueeze(1).expand_as(sequence)
        gate = self.gate(torch.cat([sequence, expanded], dim=-1))
        return sequence + gate * expanded


class PrefixReranker(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.user_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_proj = nn.Linear(hidden_dim, hidden_dim)

    def score(self, user_repr: torch.Tensor, prefix_repr: torch.Tensor) -> torch.Tensor:
        user_vec = F.normalize(self.user_proj(user_repr), dim=-1)
        path_vec = F.normalize(self.path_proj(prefix_repr), dim=-1)
        return (user_vec * path_vec).sum(dim=-1)


@dataclass
class DecodeResult:
    items: torch.Tensor
    source: List[str]
    diagnostics: Dict[str, float]


class BARGEModel(nn.Module):
    def __init__(self, num_items: int, codebook_size: int, hidden_dim: int = 64, history_len: int = 6) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.history_len = history_len
        self.item_embeddings = nn.Embedding(num_items, hidden_dim)
        self.position_embeddings = nn.Embedding(history_len, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ica = ItemContextAwareAttention(hidden_dim)
        self.reranker = PrefixReranker(hidden_dim)

        self.main_code_embeddings = nn.ModuleList([nn.Embedding(codebook_size + 1, hidden_dim) for _ in range(2)])
        self.aux_code_embeddings = nn.ModuleList([nn.Embedding(codebook_size + 1, hidden_dim) for _ in range(2)])
        self.main_decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.aux_decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.main_heads = nn.ModuleList([nn.Linear(hidden_dim, codebook_size) for _ in range(2)])
        self.aux_heads = nn.ModuleList([nn.Linear(hidden_dim, codebook_size) for _ in range(2)])
        self.item_scorer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))
        self.aux_rotation = nn.Parameter(torch.eye(hidden_dim))

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(history.size(1), device=history.device).unsqueeze(0).expand_as(history)
        sequence = self.item_embeddings(history) + self.position_embeddings(positions)
        enhanced = self.ica(sequence)
        encoded, hidden = self.encoder(enhanced)
        return hidden.squeeze(0) + encoded.mean(dim=1)

    def _decode_teacher(
        self,
        user_repr: torch.Tensor,
        targets: torch.Tensor,
        decoder: nn.GRUCell,
        heads: nn.ModuleList,
        embeddings: nn.ModuleList,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = user_repr
        prev = self.start_token.unsqueeze(0).expand(user_repr.size(0), -1)
        logits_all = []
        prefix = torch.zeros_like(user_repr)
        prefix_states = []
        for level in range(2):
            state = decoder(prev, state)
            logits = heads[level](state)
            logits_all.append(logits)
            target_code = targets[:, level]
            token_emb = embeddings[level](target_code + 1)
            prefix = prefix + token_emb
            prefix_states.append(prefix.clone())
            prev = token_emb
        return torch.stack(logits_all, dim=1), torch.stack(prefix_states, dim=1)

    def forward(self, history: torch.Tensor, main_targets: torch.Tensor, aux_targets: torch.Tensor):
        user_repr = self.encode_history(history)
        main_logits, main_prefix = self._decode_teacher(
            user_repr, main_targets, self.main_decoder, self.main_heads, self.main_code_embeddings
        )
        rotated_user = user_repr @ self._orthogonal_rotation().t()
        aux_logits, aux_prefix = self._decode_teacher(
            rotated_user, aux_targets, self.aux_decoder, self.aux_heads, self.aux_code_embeddings
        )
        return {
            "user_repr": user_repr,
            "rotated_user": rotated_user,
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "main_prefix": main_prefix,
            "aux_prefix": aux_prefix,
        }

    def _orthogonal_rotation(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.aux_rotation)
        return q

    def item_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        item_matrix = self.item_scorer(self.item_embeddings.weight)
        return user_repr @ item_matrix.t()

    def _beam_candidates(
        self,
        user_repr: torch.Tensor,
        decoder: nn.GRUCell,
        heads: nn.ModuleList,
        embeddings: nn.ModuleList,
        beam_width: int = 4,
    ):
        batch_size = user_repr.size(0)
        beams = [[(0.0, self.start_token, user_repr[row], [], torch.zeros(self.hidden_dim, device=user_repr.device))] for row in range(batch_size)]
        diagnostics: List[float] = []
        for level in range(2):
            next_beams = []
            for row in range(batch_size):
                row_candidates = []
                for score, prev, state, codes, prefix in beams[row]:
                    new_state = decoder(prev.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
                    logits = heads[level](new_state)
                    probs = torch.log_softmax(logits, dim=-1)
                    top_scores, top_ids = torch.topk(probs, beam_width)
                    for local_score, code_id in zip(top_scores.tolist(), top_ids.tolist()):
                        token_emb = embeddings[level](torch.tensor(code_id + 1, device=user_repr.device))
                        new_prefix = prefix + token_emb
                        rerank = float(self.reranker.score(user_repr[row : row + 1], new_prefix.unsqueeze(0)).item())
                        row_candidates.append((score + local_score + 0.35 * rerank, token_emb, new_state, codes + [code_id], new_prefix))
                row_candidates.sort(key=lambda item: item[0], reverse=True)
                diagnostics.append(float(row_candidates[0][0]))
                next_beams.append(row_candidates[:beam_width])
            beams = next_beams
        return beams, diagnostics

    @torch.no_grad()
    def decode_plain(self, history: torch.Tensor, catalog: Catalog) -> DecodeResult:
        user_repr = self.encode_history(history)
        state = user_repr
        prev = self.start_token.unsqueeze(0).expand(history.size(0), -1)
        codes = []
        for level in range(2):
            state = self.main_decoder(prev, state)
            logits = self.main_heads[level](state)
            code = logits.argmax(dim=-1)
            codes.append(code)
            prev = self.main_code_embeddings[level](code + 1)
        code_tensor = torch.stack(codes, dim=1)
        items = []
        for row in range(history.size(0)):
            item_id = catalog.resolve_main(code_tensor[row].tolist())
            items.append(0 if item_id is None else item_id)
        return DecodeResult(
            items=torch.tensor(items, device=history.device),
            source=["main-greedy"] * history.size(0),
            diagnostics={"avg_beam_score": 0.0},
        )

    @torch.no_grad()
    def decode_barge(self, history: torch.Tensor, catalog: Catalog, beam_width: int = 4) -> DecodeResult:
        user_repr = self.encode_history(history)
        item_bias = self.item_scores(user_repr)
        main_beams, main_diag = self._beam_candidates(user_repr, self.main_decoder, self.main_heads, self.main_code_embeddings, beam_width)
        aux_user = user_repr @ self._orthogonal_rotation().t()
        aux_beams, aux_diag = self._beam_candidates(aux_user, self.aux_decoder, self.aux_heads, self.aux_code_embeddings, beam_width)

        chosen_items = []
        sources = []
        for row in range(history.size(0)):
            seen = set(int(v) for v in history[row].tolist())
            candidates: Dict[int, Tuple[float, str]] = {}
            for score, _, _, codes, _ in main_beams[row]:
                item_id = catalog.resolve_main(codes)
                if item_id is not None and item_id not in seen:
                    fused = float(score + item_bias[row, item_id].item())
                    candidates[item_id] = max(candidates.get(item_id, (-1e9, "main")), (fused, "main"), key=lambda x: x[0])
            for score, _, _, codes, _ in aux_beams[row]:
                item_id = catalog.resolve_aux(codes)
                if item_id is not None and item_id not in seen:
                    fused = float(score + item_bias[row, item_id].item())
                    fused = float(torch.logsumexp(torch.tensor([fused, candidates.get(item_id, (-1e9, "aux"))[0]], device=history.device), dim=0).item())
                    candidates[item_id] = (fused, "dual" if item_id in candidates else "aux")
            if not candidates:
                fallback = int(self.decode_plain(history[row : row + 1], catalog).items[0].item())
                chosen_items.append(fallback)
                sources.append("fallback")
                continue
            best_item, (_, source) = max(candidates.items(), key=lambda kv: kv[1][0])
            chosen_items.append(best_item)
            sources.append(source)
        return DecodeResult(
            items=torch.tensor(chosen_items, device=history.device),
            source=sources,
            diagnostics={
                "avg_main_beam_score": float(sum(main_diag) / max(1, len(main_diag))),
                "avg_aux_beam_score": float(sum(aux_diag) / max(1, len(aux_diag))),
            },
        )


def compute_hpr_loss(user_repr: torch.Tensor, prefix_states: torch.Tensor) -> torch.Tensor:
    batch_size = user_repr.size(0)
    positives = prefix_states[:, -1]
    negatives = torch.roll(positives, shifts=1, dims=0)
    pos = F.cosine_similarity(user_repr, positives)
    neg = F.cosine_similarity(user_repr, negatives)
    logits = torch.stack([pos, neg], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=user_repr.device)
    return F.cross_entropy(logits, labels)


def training_loss(outputs, main_targets: torch.Tensor, aux_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    main_loss = F.cross_entropy(outputs["main_logits"].reshape(-1, outputs["main_logits"].size(-1)), main_targets.reshape(-1))
    aux_loss = F.cross_entropy(outputs["aux_logits"].reshape(-1, outputs["aux_logits"].size(-1)), aux_targets.reshape(-1))
    hpr_main = compute_hpr_loss(outputs["user_repr"], outputs["main_prefix"])
    hpr_aux = compute_hpr_loss(outputs["rotated_user"], outputs["aux_prefix"])
    loss = main_loss + aux_loss + 0.25 * (hpr_main + hpr_aux)
    main_acc = (outputs["main_logits"].argmax(dim=-1) == main_targets).float().mean().item()
    aux_acc = (outputs["aux_logits"].argmax(dim=-1) == aux_targets).float().mean().item()
    return loss, {"main_token_acc": main_acc, "aux_token_acc": aux_acc}
