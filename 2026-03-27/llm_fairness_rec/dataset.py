import torch
from torch.utils.data import Dataset

class RecDataset(Dataset):
    def __init__(self, data_path):
        # Mock recommendation dataset with sensitive attributes
        self.data = [{"user_id": 1, "item_ids": [10, 20], "sensitive_attr": 0, "target_item": 30}]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["user_id"], item["item_ids"], item["sensitive_attr"], item["target_item"]
