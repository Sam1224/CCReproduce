import torch
import torch.nn as nn

class SPathRAG(nn.Module):
    def __init__(self, vocab_size, latent_dim=128):
        super(SPathRAG, self).__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.path_encoder = nn.Linear(latent_dim, latent_dim)
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads=8)
        self.output_head = nn.Linear(latent_dim, vocab_size)
        
    def forward(self, question_ids, path_latents):
        # question_ids: [seq_len, batch_size]
        # path_latents: [path_len, batch_size, latent_dim]
        q_emb = self.embedding(question_ids)
        attn_out, _ = self.cross_attention(q_emb, path_latents, path_latents)
        logits = self.output_head(attn_out)
        return logits
