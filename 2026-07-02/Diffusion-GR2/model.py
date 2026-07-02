import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionGR2(nn.Module):
    """
    Diffusion-GR2: Diffusion Generative Reasoning Re-ranker.
    Converts AR re-ranker to a block-diffusion decoder for faster inference.
    """
    def __init__(self, vocab_size, hidden_dim, num_items=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        self.reasoning_head = nn.Linear(hidden_dim, vocab_size)
        self.ranking_head = nn.Linear(hidden_dim, vocab_size)
        self.num_items = num_items

    def forward(self, x, timesteps):
        # x: [batch, seq_len] noisy tokens
        # timesteps: [batch]
        h = self.embedding(x)
        # Add timestep embedding (simplified)
        h = h + timesteps.view(-1, 1, 1)
        out = self.transformer(h)
        
        # Predict both reasoning tokens and final ranking
        logits_reasoning = self.reasoning_head(out)
        logits_ranking = self.ranking_head(out[:, -self.num_items:, :])
        
        return logits_reasoning, logits_ranking

def train():
    model = DiffusionGR2(vocab_size=1000, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Toy dataset
    batch_size = 4
    seq_len = 50
    
    for epoch in range(1):
        x = torch.randint(0, 1000, (batch_size, seq_len))
        target_reasoning = torch.randint(0, 1000, (batch_size, seq_len))
        target_ranking = torch.randint(0, 1000, (batch_size, 10))
        timesteps = torch.rand(batch_size)
        
        logits_res, logits_rank = model(x, timesteps)
        
        loss = criterion(logits_res.view(-1, 1000), target_reasoning.view(-1)) + \
               criterion(logits_rank.view(-1, 1000), target_ranking.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
