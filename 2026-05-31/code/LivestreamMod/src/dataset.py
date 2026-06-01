"""
Toy dataset for LivestreamMod reproduction.
Generates synthetic multimodal features (text + audio + visual) for livestream
content moderation. Real data would come from a live e-commerce platform.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os, json, random


def generate_toy_cases(num_cases: int = 200, feat_dim: int = 128,
                        out_dir: str = "data/toy_cases", seed: int = 42):
    """Generate synthetic multimodal livestream cases with violation labels."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Violation categories: 0=normal, 1=false_claims, 2=prohibited_goods,
    # 3=misleading_pricing, 4=emerging_violation
    num_classes = 5
    cases = []
    for i in range(num_cases):
        label = rng.integers(0, num_classes)
        # Each violation class has a characteristic "cluster center"
        center = rng.standard_normal(feat_dim) * 2.0 if label == 0 \
                 else rng.standard_normal(feat_dim) + np.eye(num_classes, feat_dim)[label] * 3.0
        noise = rng.standard_normal(feat_dim) * 0.5
        # Separate features for text, audio, visual modalities
        text_feat = center + rng.standard_normal(feat_dim) * 0.3 + noise
        audio_feat = center + rng.standard_normal(feat_dim) * 0.4 + noise
        visual_feat = center + rng.standard_normal(feat_dim) * 0.5 + noise
        cases.append({
            "id": f"case_{i:04d}",
            "text_feat": text_feat.tolist(),
            "audio_feat": audio_feat.tolist(),
            "visual_feat": visual_feat.tolist(),
            "label": int(label),
            "is_violation": int(label > 0),
        })
        # Emerging violation (class 4) has fewer training examples — simulate cold-start
    with open(os.path.join(out_dir, "cases.json"), "w") as f:
        json.dump(cases, f)
    print(f"Generated {num_cases} cases → {out_dir}/cases.json")
    return cases


class LivestreamDataset(Dataset):
    """Loads toy multimodal livestream cases."""
    def __init__(self, data_path: str, split: str = "train", train_ratio: float = 0.7):
        with open(data_path) as f:
            all_cases = json.load(f)
        n = len(all_cases)
        split_idx = int(n * train_ratio)
        if split == "train":
            self.cases = all_cases[:split_idx]
        else:
            self.cases = all_cases[split_idx:]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        c = self.cases[idx]
        text = torch.tensor(c["text_feat"], dtype=torch.float32)
        audio = torch.tensor(c["audio_feat"], dtype=torch.float32)
        visual = torch.tensor(c["visual_feat"], dtype=torch.float32)
        # Fuse: simple concatenation (paper uses cross-modal attention; simplified here)
        fused = torch.cat([text, audio, visual], dim=0)
        return {
            "fused": fused,
            "text": text,
            "audio": audio,
            "visual": visual,
            "label": torch.tensor(c["label"], dtype=torch.long),
            "is_violation": torch.tensor(c["is_violation"], dtype=torch.float32),
        }


def get_loaders(data_dir: str, batch_size: int = 32):
    train_ds = LivestreamDataset(os.path.join(data_dir, "cases.json"), split="train")
    val_ds = LivestreamDataset(os.path.join(data_dir, "cases.json"), split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
