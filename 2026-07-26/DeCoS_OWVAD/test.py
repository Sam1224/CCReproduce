import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DeCoSToyDataset, DefinitionBank
from metrics import summarize_metrics
from model import CenteredMarginScorer, DeCoSScorer, DefinitionBlindScorer, MarginOnlyScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate toy OWVAD scorers.")
    parser.add_argument("--checkpoint", type=str, default="decos_toy.pt")
    parser.add_argument("--test-samples", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def collect_outputs(model, data_loader, anomaly_embeddings, normal_embedding):
    model.eval()
    all_scores = []
    all_labels = []
    all_spans = []
    with torch.no_grad():
        for batch in data_loader:
            visual_features = batch["visual_features"].float()
            result = model(visual_features, anomaly_embeddings, normal_embedding)
            all_scores.append(result["scores"])
            all_labels.append(batch["frame_labels"].long())
            all_spans.append(batch["event_spans"].long())
    return torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_spans, dim=0)


def main() -> None:
    args = parse_args()
    package = torch.load(Path(args.checkpoint), map_location="cpu")
    config = package["config"]
    bank_dict = package["definition_bank"]
    definition_bank = DefinitionBank(
        normal_embedding=bank_dict["normal_embedding"].float(),
        anomaly_embeddings=bank_dict["anomaly_embeddings"].float(),
        shared_direction=bank_dict["shared_direction"].float(),
        class_directions=bank_dict["class_directions"].float(),
    )

    test_dataset = DeCoSToyDataset(
        num_samples=args.test_samples,
        seq_len=config["seq_len"],
        feature_dim=config["feature_dim"],
        num_classes=config["num_classes"],
        definition_bank=definition_bank,
        seed=args.seed,
    )
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    models = {
        "definition_blind": DefinitionBlindScorer(definition_bank.shared_direction),
        "margin_only": MarginOnlyScorer(definition_bank.shared_direction),
        "centered_margin": CenteredMarginScorer(definition_bank.shared_direction),
        "decos": DeCoSScorer(
            feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            shared_direction=definition_bank.shared_direction,
        ),
    }
    models["decos"].load_state_dict(package["model_state"])

    print("name                 std_auroc  dc_disc  dc_det_delta  dc_sel_delta")
    print("-" * 70)
    for name, model in models.items():
        scores, labels, spans = collect_outputs(
            model,
            data_loader,
            definition_bank.anomaly_embeddings,
            definition_bank.normal_embedding,
        )
        metrics = summarize_metrics(scores, labels, spans, config["num_classes"])
        print(
            f"{name:20s} {metrics['std_auroc']:.4f}     {metrics['dc_disc']:.4f}   {metrics['dc_det_delta']:.4f}        {metrics['dc_sel_delta']:.4f}"
        )

    debug_scores, _, debug_spans = collect_outputs(
        models["decos"],
        data_loader,
        definition_bank.anomaly_embeddings,
        definition_bank.normal_embedding,
    )
    first_multi_event = next(index for index in range(debug_spans.size(0)) if (debug_spans[index, 1, 0] >= 0).item())
    spans = debug_spans[first_multi_event]
    score_matrix = debug_scores[first_multi_event]
    print("\ndebug_multi_event=")
    print(spans)
    for span in spans:
        start, end, label = span.tolist()
        if start < 0:
            continue
        mean_score = score_matrix[start:end, label - 1].mean().item()
        print(f"query_label={label} span=({start}, {end}) mean_score={mean_score:.4f}")


if __name__ == "__main__":
    main()
