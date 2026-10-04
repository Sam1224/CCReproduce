import argparse

import torch
import torch.nn.functional as F

from data import FeatureConfig, ToyLivestreamDataset, build_reference_bank
from model import ModelConfig, MultimodalStudent, FrozenMLLMTeacher, hybrid_decision


POLICY_LABELS = ["clean", "adult", "illegal_goods", "misleading_commerce", "harassment"]


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    data_config = FeatureConfig(**checkpoint.get("data_config", {}))
    model_config = ModelConfig(**checkpoint.get("config", {}))
    model = MultimodalStudent(model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = ToyLivestreamDataset(size=32, config=data_config)
    sample = dataset[args.index]
    batch = {key: value.unsqueeze(0) for key, value in sample.items() if key in {"visual", "audio", "text"}}
    teacher = FrozenMLLMTeacher(model_config)
    reference_bank = build_reference_bank(dataset)
    ref_batch = {
        "visual": reference_bank["raw"][:, :data_config.visual_dim],
        "audio": reference_bank["raw"][:, data_config.visual_dim:data_config.visual_dim + data_config.audio_dim],
        "text": reference_bank["raw"][:, data_config.visual_dim + data_config.audio_dim:],
    }
    with torch.no_grad():
        out = model(batch)
        probs = F.softmax(out["logits"], dim=-1)
        refs = teacher(ref_batch)["embedding"]
        ref_scores = model.score_references(out["embedding"], refs)
        decision = hybrid_decision(probs, ref_scores).item()
        pred_label = int(probs.argmax(dim=-1).item())
        top_ref = int(ref_scores.argmax(dim=-1).item())
    print({
        "decision": "flag" if decision else "allow",
        "predicted_policy": POLICY_LABELS[pred_label],
        "policy_confidence": round(float(probs.max().item()), 4),
        "nearest_reference_id": top_ref,
        "reference_score": round(float(torch.sigmoid(ref_scores.max()).item()), 4),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    main(parser.parse_args())
