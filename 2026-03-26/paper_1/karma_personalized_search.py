import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper: KARMA: Knowledge-Action Regularized Multimodal Alignment for Personalized Search at Taobao
Summary: A unified framework closing the Knowledge-Action gap in LLM-based personalized search.
Core: Next-interest embedding with semantic decodability via generation and reconstruction objectives.
"""

class KARMAModel(nn.Module):
    def __init__(self, d_model=512, vocab_size=32000, history_len=20):
        super(KARMAModel, self).__init__()
        # Multimodal Alignment: text, image, and behavior features
        self.behavior_encoder = nn.Linear(d_model, d_model)
        self.text_encoder = nn.Linear(d_model, d_model)
        self.image_encoder = nn.Linear(d_model, d_model)
        
        # Unified History Compressor (Interest Embedding)
        self.interest_compressor = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.interest_proj = nn.Linear(d_model, d_model)
        
        # Action Head (Ranking/Retrieval)
        self.action_head = nn.Linear(d_model, 1)
        
        # Auxiliary Objective 1: Semantic Generation Decoder
        self.semantic_decoder = nn.Linear(d_model, vocab_size)
        
        # Auxiliary Objective 2: Embedding-Conditioned Reconstruction
        self.reconstructor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, behavior_seq, text_seq, image_seq, target_action=None, target_text=None):
        """
        Args:
            behavior_seq: (batch, seq, d_model)
            text_seq: (batch, seq, d_model)
            image_seq: (batch, seq, d_model)
        """
        # 1. Multimodal Feature Alignment
        b_feat = self.behavior_encoder(behavior_seq)
        t_feat = self.text_encoder(text_seq)
        i_feat = self.image_encoder(image_seq)
        
        # Simple fusion (weighted sum or concat)
        fused_history = b_feat + t_feat + i_feat # (batch, seq, d_model)
        
        # 2. History Compression into Interest Embedding
        compressed = self.interest_compressor(fused_history.transpose(0, 1)) # (seq, batch, d_model)
        interest_emb = compressed.mean(dim=0) # (batch, d_model)
        interest_emb = self.interest_proj(interest_emb)
        
        # 3. Action Logic (Retrieval/Ranking)
        action_score = self.action_head(interest_emb)
        
        # 4. Auxiliary Decodability
        # Generation: history-conditioned semantic generation
        gen_logits = self.semantic_decoder(interest_emb) # (batch, vocab_size)
        
        # Reconstruction: embedding-conditioned reconstruction of original semantics
        recon_history = self.reconstructor(interest_emb)
        
        return action_score, gen_logits, recon_history

def karma_loss_fn(action_score, target_action, gen_logits, target_text_idx, recon_history, original_semantics, alpha=0.5, beta=0.5):
    # Action Loss (e.g., Cross Entropy or MSE for CTR)
    action_loss = F.mse_loss(action_score, target_action)
    
    # Knowledge Regularization 1: Semantic Generation Loss
    gen_loss = F.cross_entropy(gen_logits, target_text_idx)
    
    # Knowledge Regularization 2: Reconstruction Loss
    recon_loss = F.mse_loss(recon_history, original_semantics)
    
    # Total loss: Action + Knowledge (Regularized)
    total_loss = action_loss + alpha * gen_loss + beta * recon_loss
    return total_loss

# Example instantiation
if __name__ == "__main__":
    model = KARMAModel()
    batch_size = 4
    d_model = 512
    history_len = 10
    
    b = torch.randn(batch_size, history_len, d_model)
    t = torch.randn(batch_size, history_len, d_model)
    i = torch.randn(batch_size, history_len, d_model)
    
    score, gen, recon = model(b, t, i)
    print(f"Action score shape: {score.shape}")
    print("KARMA reproduction structure complete.")
