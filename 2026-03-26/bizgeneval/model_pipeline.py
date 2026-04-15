import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BizGenDataset(Dataset):
    """
    Mock dataset for BizGenEval tasks: slide, chart, webpage, poster, scientific figure.
    Includes capability dimensions: text rendering, layout control, attribute binding, reasoning.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # Mock feature: 128-dim prompt embedding
        self.prompts = torch.randn(num_samples, 128)
        # Mock targets: layout (4 coords), text_valid (1), attribute_match (1), reasoning_score (1)
        self.targets = torch.randn(num_samples, 7)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

class CommercialGenModel(nn.Module):
    """
    Multi-task model architecture for Commercial Visual Content Generation.
    Inspired by BizGenEval capability dimensions.
    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super(CommercialGenModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-task heads
        self.layout_head = nn.Linear(hidden_dim, 4) # x, y, w, h
        self.text_head = nn.Linear(hidden_dim, 1)   # Text rendering quality
        self.attr_head = nn.Linear(hidden_dim, 1)   # Attribute binding
        self.reason_head = nn.Linear(hidden_dim, 1) # Knowledge reasoning

    def forward(self, x):
        features = self.encoder(x)
        layout = self.layout_head(features)
        text = torch.sigmoid(self.text_head(features))
        attr = torch.sigmoid(self.attr_head(features))
        reason = self.reason_head(features)
        return layout, text, attr, reason

if __name__ == "__main__":
    dataset = BizGenDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CommercialGenModel()
    
    for prompts, targets in dataloader:
        layout, text, attr, reason = model(prompts)
        print(f"Batch layout shape: {layout.shape}")
        break
