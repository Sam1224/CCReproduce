import argparse
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticReTokenConfig, SyntheticVisualHaystackDataset
from model import ReTokenRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ReToken reproduction.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/retoken.pt")
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def evaluate() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticVisualHaystackDataset(SyntheticReTokenConfig(num_samples=256, seed=17))
    loader = DataLoader(dataset, batch_size=64)
    model = ReTokenRetriever().to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    except FileNotFoundError:
        print("checkpoint not found; evaluating randomly initialized ReToken")
    model.eval()
    correct = 0
    total = 0
    recall_hits = 0
    with torch.no_grad():
        for batch in loader:
            frame_features = batch["frame_features"].to(device)
            question_features = batch["question_features"].to(device)
            labels = batch["labels"].to(device)
            scores = model(frame_features, question_features)["scores"]
            selected = scores.topk(k=args.top_k, dim=1).indices
            relevant = torch.gather(labels, dim=1, index=selected).sum(dim=1) > 0
            recall_hits += relevant.sum().item()
            correct += relevant.sum().item()
            total += labels.size(0)
    print({"top_k": args.top_k, "retrieval_recall": recall_hits / total, "toy_answer_accuracy": correct / total})


if __name__ == "__main__":
    evaluate()
