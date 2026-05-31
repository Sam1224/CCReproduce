"""
Evaluation script for Thinking Broad, Acting Fast (arXiv: 2601.21611)
Evaluates student model on e-commerce relevance benchmark.

Metrics: NDCG@10, MAP, AUC, Accuracy
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
from torch.utils.data import DataLoader
from model import StudentRelevanceModel
from train import EComRelevanceDataset


def evaluate_relevance_model(
    model: StudentRelevanceModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate student model on relevance task.
    Returns dict of metrics: AUC, Accuracy, NDCG@10
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            scores, _ = model(
                batch["query_input_ids"].to(device),
                batch["query_attention_mask"].to(device),
                batch["product_input_ids"].to(device),
                batch["product_attention_mask"].to(device),
            )
            probs = torch.sigmoid(scores).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    accuracy = ((all_scores > 0.5) == all_labels).mean()
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5

    # NDCG@10 (treat as ranking problem within a query group)
    # For simplicity: treat all examples as one group
    ndcg = ndcg_score(all_labels.reshape(1, -1), all_scores.reshape(1, -1), k=10)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "ndcg@10": ndcg,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentRelevanceModel()
    model.load_state_dict(torch.load("checkpoints/student_model.pt", map_location=device))
    model.to(device)

    test_dataset = EComRelevanceDataset("data/ecom_relevance_test.jsonl", tokenizer=None)
    test_loader = DataLoader(test_dataset, batch_size=64)

    metrics = evaluate_relevance_model(model, test_loader, device)
    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
