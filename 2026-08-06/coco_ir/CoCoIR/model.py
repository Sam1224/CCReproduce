import torch
from torch import nn
import torch.nn.functional as functional


class TIEModel(nn.Module):
    def __init__(self, image_dim=32, text_dim=32, hidden_dim=96):
        super().__init__()
        self.image_encoder = nn.Linear(image_dim, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.history_gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.tie_projection = nn.Linear(hidden_dim, image_dim)
        self.candidate_projection = nn.Linear(image_dim, image_dim)

    def forward(self, source_image, instructions, candidates):
        batch_size, turn_count, candidate_count, image_dim = candidates.shape
        state = torch.tanh(self.image_encoder(source_image))
        logits_per_turn = []
        for turn_index in range(turn_count):
            text_state = torch.tanh(self.text_encoder(instructions[:, turn_index]))
            turn_input = torch.cat([state, text_state], dim=-1)
            state = self.history_gru(turn_input, state)
            tie_embedding = functional.normalize(self.tie_projection(state), dim=-1)
            candidate_embedding = functional.normalize(self.candidate_projection(candidates[:, turn_index]), dim=-1)
            logits = torch.einsum("bd,bnd->bn", tie_embedding, candidate_embedding) * image_dim ** 0.5
            logits_per_turn.append(logits)
            retrieved_image = torch.sum(torch.softmax(logits, dim=-1).unsqueeze(-1) * candidates[:, turn_index], dim=1)
            state = torch.tanh(self.image_encoder(retrieved_image)) + state
        return torch.stack(logits_per_turn, dim=1)


def coco_ir_loss(logits, labels):
    return functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
