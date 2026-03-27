import torch
from torch.utils.data import Dataset

class KGQADataset(Dataset):
    def __init__(self, data_path):
        # Mock data for S-Path-RAG
        self.data = [{"question": "Who is the director of Inception?", "ans": "Christopher Nolan", "kg_context": "path_latents"}]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["kg_context"], item["ans"]
