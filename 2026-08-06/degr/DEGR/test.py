import torch
from torch.utils.data import DataLoader

from dataset import ToyRerankDataset
from model import MMoEReward, DEGRGenerator


def recall_at_k(prediction, target, top_k=10):
    hits = []
    for predicted_items, target_items in zip(prediction, target):
        hits.append(len(set(predicted_items[:top_k].tolist()) & set(target_items[:top_k].tolist())) / top_k)
    return sum(hits) / len(hits)


def main():
    dataset = ToyRerankDataset(num_requests=64, seed=99)
    loader = DataLoader(dataset, batch_size=32)
    reward_model = MMoEReward(feature_dim=24)
    generator = DEGRGenerator(feature_dim=24)
    checkpoint = torch.load("degr_toy.pt", map_location="cpu")
    reward_model.load_state_dict(checkpoint["reward_model"])
    generator.load_state_dict(checkpoint["generator"])
    scores = []
    with torch.no_grad():
        for batch in loader:
            click_score, purchase_score, explore_score = reward_model(batch["features"])
            prediction = generator(batch["features"], click_score, purchase_score, explore_score)
            scores.append(recall_at_k(prediction, batch["target_order"]))
    print({"recall@10": sum(scores) / len(scores)})


if __name__ == "__main__":
    main()
