from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ACTIONS, ATTRIBUTES, VALUES


class ShopR1Policy(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 96, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.action_head = nn.Linear(hidden_size, len(ACTIONS))
        self.attribute_head = nn.Linear(hidden_size, len(ATTRIBUTES))
        self.value_head = nn.Linear(hidden_size, len(VALUES))
        self.rationale_decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.rationale_head = nn.Linear(hidden_size, vocab_size)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.pad_id).float().unsqueeze(-1)
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)

    def forward(self, input_ids: torch.Tensor, rationale_ids: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        state = self.encode(input_ids)
        outputs = {
            "action_logits": self.action_head(state),
            "attribute_logits": self.attribute_head(state),
            "value_logits": self.value_head(state),
        }
        if rationale_ids is not None:
            rationale_emb = self.embedding(rationale_ids)
            init = state.unsqueeze(0)
            decoded, _ = self.rationale_decoder(rationale_emb, init)
            outputs["rationale_logits"] = self.rationale_head(decoded)
        return outputs

    def supervised_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["rationale_ids"])
        action_loss = F.cross_entropy(outputs["action_logits"], batch["action_type"])
        attr_loss = F.cross_entropy(outputs["attribute_logits"], batch["attribute"])
        value_loss = F.cross_entropy(outputs["value_logits"], batch["value"])
        rationales = batch["rationale_ids"]
        rationale_loss = F.cross_entropy(
            outputs["rationale_logits"].reshape(-1, outputs["rationale_logits"].size(-1)),
            rationales.reshape(-1),
            ignore_index=self.pad_id,
        )
        return action_loss + 0.7 * attr_loss + 0.7 * value_loss + 0.3 * rationale_loss

    def reward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_correct = outputs["action_logits"].argmax(dim=-1).eq(batch["action_type"]).float()
        attr_correct = outputs["attribute_logits"].argmax(dim=-1).eq(batch["attribute"]).float()
        value_correct = outputs["value_logits"].argmax(dim=-1).eq(batch["value"]).float()
        format_reward = torch.ones_like(action_correct) * 0.1
        action_reward = 0.45 * action_correct
        sub_action_reward = 0.25 * attr_correct + 0.20 * value_correct
        return (format_reward + action_reward + sub_action_reward) * batch["difficulty"]

    def grpo_style_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self(batch["input_ids"])
        action_logp = F.log_softmax(outputs["action_logits"], dim=-1).gather(1, batch["action_type"].unsqueeze(1)).squeeze(1)
        attr_logp = F.log_softmax(outputs["attribute_logits"], dim=-1).gather(1, batch["attribute"].unsqueeze(1)).squeeze(1)
        value_logp = F.log_softmax(outputs["value_logits"], dim=-1).gather(1, batch["value"].unsqueeze(1)).squeeze(1)
        rewards = self.reward(batch, outputs).detach()
        advantage = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-4)
        return -(advantage * (action_logp + 0.5 * attr_logp + 0.5 * value_logp)).mean()

    def predict(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self(input_ids)
        return {key.replace("_logits", ""): value.argmax(dim=-1) for key, value in outputs.items()}
