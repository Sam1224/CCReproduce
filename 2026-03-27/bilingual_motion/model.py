import torch
import torch.nn as nn

class BiMD(nn.Module):
    def __init__(self, text_dim, motion_dim):
        super(BiMD, self).__init__()
        self.en_encoder = nn.Linear(text_dim, 256)
        self.zh_encoder = nn.Linear(text_dim, 256)
        self.cla = nn.Linear(256, 256)  # Cross-Lingual Alignment
        self.diffusion_model = nn.Sequential(
            nn.Linear(256 + motion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, motion_dim)
        )
        
    def forward(self, en_emb, zh_emb, motion_t, t):
        # CLA Strategy
        en_aligned = self.cla(self.en_encoder(en_emb))
        zh_aligned = self.cla(self.zh_encoder(zh_emb))
        
        # Diffusion reverse step
        # Assuming we take the mean or one of them for conditioning
        cond = (en_aligned + zh_aligned) / 2
        return self.diffusion_model(torch.cat([cond, motion_t], dim=-1))
