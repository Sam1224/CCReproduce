import torch
from torch.utils.data import Dataset


class ToyRerankDataset(Dataset):
    def __init__(self, num_requests=512, num_candidates=30, feature_dim=24, slate_size=10, seed=13):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_requests, num_candidates, feature_dim, generator=generator)
        click_logit = self.features[..., 0] + 0.5 * self.features[..., 3] - 0.2 * self.features[..., 5]
        purchase_logit = self.features[..., 1] + 0.3 * self.features[..., 4] + 0.2 * self.features[..., 6]
        novelty = torch.sigmoid(self.features[..., 2])
        self.click = torch.bernoulli(torch.sigmoid(click_logit), generator=generator)
        self.purchase = torch.bernoulli(torch.sigmoid(purchase_logit), generator=generator)
        base_score = 0.7 * torch.sigmoid(click_logit) + 0.3 * novelty
        self.target_order = torch.argsort(base_score, dim=1, descending=True)[:, :slate_size]
        target_items = torch.gather(base_score, 1, self.target_order)
        self.sequence_label = (target_items.mean(dim=1) > target_items.mean()).float()
        self.scroll = torch.bernoulli(torch.clamp(novelty.mean(dim=1), 0.05, 0.95), generator=generator)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "click": self.click[index],
            "purchase": self.purchase[index],
            "target_order": self.target_order[index],
            "sequence_label": self.sequence_label[index],
            "scroll": self.scroll[index],
        }
