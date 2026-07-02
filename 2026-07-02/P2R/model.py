import torch
import torch.nn as nn

class Perceiver(nn.Module):
    def __init__(self, visual_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(visual_dim, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )
        self.bbox_head = nn.Linear(64 * 49, 4) # Predict [x, y, w, h]

    def forward(self, img):
        return torch.sigmoid(self.bbox_head(self.conv(img)))

class Reasoner(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.visual_proj = nn.Linear(256, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=4
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text, visual_features):
        tgt = self.embedding(text)
        mem = self.visual_proj(visual_features)
        out = self.decoder(tgt, mem)
        return self.head(out)

class P2RModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceiver = Perceiver(3, 128)
        self.reasoner = Reasoner(1000, 256)

    def forward(self, img, text):
        # Stage 1: Perceive
        bbox = self.perceiver(img)
        # In practice, crop and extract features from bbox
        # Here we simulate visual features extraction
        visual_features = torch.randn(img.size(0), 10, 256) 
        
        # Stage 2: Reason
        logits = self.reasoner(text, visual_features)
        return bbox, logits

if __name__ == "__main__":
    model = P2RModel()
    img = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 1000, (2, 20))
    bbox, logits = model(img, text)
    print("BBox shape:", bbox.shape)
    print("Logits shape:", logits.shape)
