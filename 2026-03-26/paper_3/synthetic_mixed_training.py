import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper: Synthetic Mixed Training: Scaling Parametric Knowledge Acquisition Beyond RAG
Summary: Scaling parametric knowledge acquisition with synthetic data (QA + Docs).
Core: Jointly training on synthetic QA and synthetic documents (Focal Rewriting).
"""

class SyntheticMixedTrainer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512):
        super(SyntheticMixedTrainer, self).__init__()
        # Large Language Model (e.g., Transformer-based)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8), num_layers=6)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Focal Rewriting Module (Teacher/Generator)
        # In reality, this would be a separate LLM generating synthetic docs
        self.focal_rewriter = nn.Linear(d_model, d_model) # Simulates rewriting docs conditioned on QA

    def focal_rewriting_logic(self, query_emb, original_doc_emb):
        """
        Focal Rewriting: Question-conditioned document generation.
        Condition the generator on the question (QA pair) to target knowledge gaps.
        """
        # Question-conditioned document embedding
        conditioned_doc_emb = original_doc_emb + self.focal_rewriter(query_emb)
        return conditioned_doc_emb

    def forward(self, qa_input_ids, doc_input_ids):
        """
        Forward pass for Synthetic Mixed Training.
        Jointly training on synthetic QA pairs and synthetic documents.
        """
        # 1. Processing QA Pairs (Synthetic QA)
        qa_hidden = self.encoder(qa_input_ids.transpose(0, 1))
        qa_logits = self.lm_head(qa_hidden.transpose(0, 1))
        
        # 2. Processing Synthetic Documents (Focal Rewriting results)
        doc_hidden = self.encoder(doc_input_ids.transpose(0, 1))
        doc_logits = self.lm_head(doc_hidden.transpose(0, 1))
        
        return qa_logits, doc_logits

def mixed_training_loss(qa_logits, qa_labels, doc_logits, doc_labels, lambda_qa=0.5, lambda_doc=0.5):
    # MLE Loss for QA Pairs
    qa_loss = F.cross_entropy(qa_logits.view(-1, qa_logits.size(-1)), qa_labels.view(-1))
    
    # MLE Loss for Synthetic Documents (Knowledge Acquisition)
    doc_loss = F.cross_entropy(doc_logits.view(-1, doc_logits.size(-1)), doc_labels.view(-1))
    
    # Combined Loss (Restores Scaling Behavior)
    total_loss = lambda_qa * qa_loss + lambda_doc * doc_loss
    return total_loss

# Example Usage (Training Loop Pseudo-code)
def train_synthetic_mixed_model():
    """
    Synthetic Mixed Training Algorithm:
    1. Generate Synthetic QA pairs via strong teacher.
    2. Apply Focal Rewriting: Question-conditioned document generation.
    3. Jointly train 8B model on both QA and Docs.
    4. Observe log-linear scaling as synthetic data size increases.
    """
    model = SyntheticMixedTrainer()
    print("Synthetic Mixed Training model structure initialized.")
    print("Joint training on QA + Docs restores scaling beyond RAG.")

if __name__ == "__main__":
    train_synthetic_mixed_model()
