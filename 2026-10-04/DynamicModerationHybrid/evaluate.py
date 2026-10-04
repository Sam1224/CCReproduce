import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from data import FeatureConfig, ToyLivestreamDataset, build_reference_bank
from model import ModelConfig, MultimodalStudent, FrozenMLLMTeacher, hybrid_decision


def recall_at_precision(labels, scores, target_precision=0.8):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    valid = recall[precision >= target_precision]
    return float(valid.max()) if len(valid) else 0.0


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    data_config = FeatureConfig(**checkpoint.get("data_config", {}))
    model_config = ModelConfig(**checkpoint.get("config", {}))
    dataset = ToyLivestreamDataset(size=args.samples, config=data_config)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = MultimodalStudent(model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    teacher = FrozenMLLMTeacher(model_config)
    reference_bank = build_reference_bank(dataset)
    ref_batch = {
        "visual": reference_bank["raw"][:, :data_config.visual_dim],
        "audio": reference_bank["raw"][:, data_config.visual_dim:data_config.visual_dim + data_config.audio_dim],
        "text": reference_bank["raw"][:, data_config.visual_dim + data_config.audio_dim:],
    }
    with torch.no_grad():
        reference_embeddings = teacher(ref_batch)["embedding"]

    all_labels, cls_scores, sim_scores, hybrid_scores = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            probs = F.softmax(out["logits"], dim=-1)
            scores = model.score_references(out["embedding"], reference_embeddings)
            cls = 1.0 - probs[:, 0]
            sim = torch.sigmoid(scores.max(dim=1).values)
            hybrid = torch.maximum(cls, sim)
            all_labels.extend((batch["label"] > 0).int().tolist())
            cls_scores.extend(cls.tolist())
            sim_scores.extend(sim.tolist())
            hybrid_scores.extend(hybrid.tolist())

    labels = torch.tensor(all_labels).numpy()
    print({
        "classification_recall_at_p80": recall_at_precision(labels, cls_scores, args.target_precision),
        "similarity_recall_at_p80": recall_at_precision(labels, sim_scores, args.target_precision),
        "hybrid_recall_at_p80": recall_at_precision(labels, hybrid_scores, args.target_precision),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-precision", type=float, default=0.8)
    main(parser.parse_args())
