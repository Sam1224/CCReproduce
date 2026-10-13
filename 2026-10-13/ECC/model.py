from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class Tokenizer:
    def __init__(self, texts: Sequence[str]):
        vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        for text in texts:
            for token in text.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.token_to_id = vocab
        self.id_to_token = {value: key for key, value in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.token_to_id[BOS_TOKEN])
        for token in text.split():
            tokens.append(self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]))
        if add_eos:
            tokens.append(self.token_to_id[EOS_TOKEN])
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)
            if token in (PAD_TOKEN, BOS_TOKEN):
                continue
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        return " ".join(tokens)


def pad_batch(sequences: Sequence[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = []
    for seq in sequences:
        padded = seq + [pad_id] * (max_len - len(seq))
        batch.append(padded)
    return torch.tensor(batch, dtype=torch.long)


class SuggestionSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_input_ids: torch.Tensor) -> torch.Tensor:
        src_embed = self.embedding(src_ids)
        _, hidden = self.encoder(src_embed)
        tgt_embed = self.embedding(tgt_input_ids)
        decoder_output, _ = self.decoder(tgt_embed, hidden)
        return self.output(decoder_output)

    def sequence_log_prob(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(src_ids, tgt_ids[:, :-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        target = tgt_ids[:, 1:].unsqueeze(-1)
        token_scores = torch.gather(log_probs, dim=-1, index=target).squeeze(-1)
        mask = (tgt_ids[:, 1:] != 0).float()
        return (token_scores * mask).sum(dim=1)

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32) -> torch.Tensor:
        src_embed = self.embedding(src_ids)
        _, hidden = self.encoder(src_embed)
        current = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=src_ids.device)
        outputs = []
        for _ in range(max_len):
            embed = self.embedding(current[:, -1:])
            decoder_output, hidden = self.decoder(embed, hidden)
            logits = self.output(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(next_token)
            current = torch.cat([current, next_token], dim=1)
            if torch.all(next_token.squeeze(-1) == eos_id):
                break
        return torch.cat(outputs, dim=1)


def build_sft_batch(
    rows: Sequence[Tuple[str, str]],
    tokenizer: Tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_ids = [tokenizer.encode(prompt, add_bos=False, add_eos=True) for prompt, _ in rows]
    tgt_ids = [tokenizer.encode(target, add_bos=True, add_eos=True) for _, target in rows]
    return pad_batch(src_ids, 0), pad_batch(tgt_ids, 0)


def build_preference_batch(
    rows: Sequence[Tuple[str, str, str, str]],
    tokenizer: Tokenizer,
) -> Dict[str, torch.Tensor]:
    src_ids = [tokenizer.encode(prompt, add_bos=False, add_eos=True) for prompt, _, _, _ in rows]
    chosen_ids = [tokenizer.encode(chosen, add_bos=True, add_eos=True) for _, chosen, _, _ in rows]
    shuffled_ids = [tokenizer.encode(shuffled, add_bos=True, add_eos=True) for _, _, shuffled, _ in rows]
    rejected_ids = [tokenizer.encode(rejected, add_bos=True, add_eos=True) for _, _, _, rejected in rows]
    return {
        "src_ids": pad_batch(src_ids, 0),
        "chosen_ids": pad_batch(chosen_ids, 0),
        "shuffled_ids": pad_batch(shuffled_ids, 0),
        "rejected_ids": pad_batch(rejected_ids, 0),
    }


def cpo_loss(
    model: SuggestionSeq2Seq,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.5,
    consistency_weight: float = 0.2,
) -> torch.Tensor:
    src_ids = batch["src_ids"]
    chosen_score = model.sequence_log_prob(src_ids, batch["chosen_ids"])
    shuffled_score = model.sequence_log_prob(src_ids, batch["shuffled_ids"])
    rejected_score = model.sequence_log_prob(src_ids, batch["rejected_ids"])
    preference = -torch.log(torch.sigmoid(beta * (chosen_score - rejected_score)) + 1e-8).mean()
    consistency = ((chosen_score - shuffled_score) ** 2).mean()
    return preference + consistency_weight * consistency


def exact_match_accuracy(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> float:
    if not predictions:
        return 0.0
    matched = 0
    for prediction, target in zip(predictions, targets):
        if prediction.strip() == target.strip():
            matched += 1
    return float(matched) / float(len(predictions))
