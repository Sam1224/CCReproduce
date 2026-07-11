import torch
from torch.utils.data import Dataset


class ToyMarketplaceUpliftDataset(Dataset):
    """Synthetic multi-seller coupon data aligned with CanniUplift inputs."""

    def __init__(self, length: int = 2048, num_sellers: int = 6, feature_dim: int = 16):
        self.length = length
        self.num_sellers = num_sellers
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        user = torch.randn(self.feature_dim, generator=generator)
        seller_features = torch.randn(self.num_sellers, self.feature_dim, generator=generator)
        treatment = torch.randint(0, 2, (self.num_sellers,), generator=generator).float()
        coupon_value = treatment * torch.rand(self.num_sellers, generator=generator)
        price = 20 + 80 * torch.rand(self.num_sellers, generator=generator)
        same_category_visits = torch.floor(torch.rand(self.num_sellers, generator=generator) * 5.0)
        high_price_ratio = torch.rand(self.num_sellers, generator=generator) * 2.0

        affinity = (seller_features @ user) / (self.feature_dim ** 0.5)
        base_gmv = torch.relu(affinity + 0.2) * price / 100.0
        seller_cannibalization = 0.10 * same_category_visits + 0.15 * torch.relu(high_price_ratio - 1.0)
        true_seller_uplift = treatment * coupon_value * torch.sigmoid(affinity) * (1.0 - 0.35 * seller_cannibalization)
        organic_purchase = torch.sigmoid(affinity + 0.4 * high_price_ratio)
        redemption_prob = torch.sigmoid(1.5 * coupon_value - 0.8 * organic_purchase)
        redemption = torch.bernoulli(redemption_prob * treatment)
        noise = 0.02 * torch.randn(self.num_sellers, generator=generator)
        observed_gmv = base_gmv + true_seller_uplift + noise
        platform_delta = true_seller_uplift.sum() - 0.25 * (seller_cannibalization * treatment).sum()
        return {
            "user": user,
            "seller_features": seller_features,
            "treatment": treatment,
            "coupon_value": coupon_value,
            "redemption": redemption,
            "observed_gmv": observed_gmv,
            "seller_uplift": true_seller_uplift,
            "platform_delta": platform_delta.unsqueeze(0),
        }
