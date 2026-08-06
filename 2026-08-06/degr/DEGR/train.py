import torch
from torch.utils.data import DataLoader
import torch.nn.functional as functional

from dataset import ToyRerankDataset
from model import MMoEReward, DEGRGenerator, adaptive_orpo_loss


def train(epochs=3, batch_size=32, learning_rate=1e-3):
    dataset = ToyRerankDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    reward_model = MMoEReward(feature_dim=24)
    generator = DEGRGenerator(feature_dim=24)
    optimizer = torch.optim.AdamW(list(reward_model.parameters()) + list(generator.parameters()), lr=learning_rate)
    for epoch_index in range(epochs):
        total_loss = 0.0
        for batch in loader:
            click_score, purchase_score, explore_score = reward_model(batch["features"])
            reward_loss = functional.binary_cross_entropy_with_logits(click_score, batch["click"]) + functional.binary_cross_entropy_with_logits(purchase_score, batch["purchase"]) + functional.binary_cross_entropy_with_logits(explore_score, batch["sequence_label"])
            policy_logits, supervised_loss, diversity_loss = generator(batch["features"], click_score, purchase_score, explore_score, batch["target_order"])
            preference_loss = adaptive_orpo_loss(policy_logits, batch["target_order"], explore_score)
            loss = reward_loss + 2.0 * supervised_loss + 0.01 * diversity_loss + preference_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch_index + 1} loss={total_loss / len(loader):.4f}")
    torch.save({"reward_model": reward_model.state_dict(), "generator": generator.state_dict()}, "degr_toy.pt")


if __name__ == "__main__":
    train()
