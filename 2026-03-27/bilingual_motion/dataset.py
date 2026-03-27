import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, data_path):
        # Mock bilingual text-to-motion data
        self.data = [{"en_text": "A person walks forward.", "zh_text": "一个人向前走。", "motion": torch.randn(20, 263)}]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["en_text"], item["zh_text"], item["motion"]
