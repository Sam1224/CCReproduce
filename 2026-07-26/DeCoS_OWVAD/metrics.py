from __future__ import annotations

import torch


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.flatten().float()
    labels = labels.flatten().bool()
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float() * 0.5
    return float((comparisons + ties).mean().item())


def standard_vad_auroc(scores: torch.Tensor, frame_labels: torch.Tensor) -> float:
    broad_scores = scores.max(dim=-1).values
    broad_labels = frame_labels > 0
    return binary_auroc(broad_scores, broad_labels)


def dc_disc_auroc(scores: torch.Tensor, frame_labels: torch.Tensor, num_classes: int) -> float:
    per_class = []
    for class_index in range(num_classes):
        positives = frame_labels == (class_index + 1)
        negatives = (frame_labels > 0) & (frame_labels != (class_index + 1))
        mask = positives | negatives
        if mask.sum() == 0:
            continue
        per_class.append(binary_auroc(scores[..., class_index][mask], positives[mask]))
    return float(sum(per_class) / max(len(per_class), 1))


def dc_det_delta(scores: torch.Tensor, frame_labels: torch.Tensor, num_classes: int) -> float:
    deltas = []
    normal_mask = frame_labels == 0
    for class_index in range(num_classes):
        positives = frame_labels == (class_index + 1)
        matched_mask = positives | normal_mask
        matched = binary_auroc(scores[..., class_index][matched_mask], positives[matched_mask])

        unmatched_scores = []
        for other_index in range(num_classes):
            if other_index == class_index:
                continue
            unmatched_scores.append(binary_auroc(scores[..., other_index][matched_mask], positives[matched_mask]))
        if unmatched_scores:
            deltas.append(matched - sum(unmatched_scores) / len(unmatched_scores))
    return float(sum(deltas) / max(len(deltas), 1))


def dc_sel_delta(scores: torch.Tensor, event_spans: torch.Tensor) -> float:
    selection_gains = []
    for sample_scores, sample_spans in zip(scores, event_spans):
        valid_spans = sample_spans[sample_spans[:, 0] >= 0]
        if valid_spans.size(0) != 2:
            continue

        (a_start, a_end, a_label), (b_start, b_end, b_label) = valid_spans.tolist()
        query_a = sample_scores[a_start:a_end, a_label - 1].mean() - sample_scores[b_start:b_end, a_label - 1].mean()
        query_b = sample_scores[b_start:b_end, b_label - 1].mean() - sample_scores[a_start:a_end, b_label - 1].mean()
        selection_gains.append(float(0.5 * (query_a + query_b)))

    return float(sum(selection_gains) / max(len(selection_gains), 1))


def summarize_metrics(scores: torch.Tensor, frame_labels: torch.Tensor, event_spans: torch.Tensor, num_classes: int) -> dict[str, float]:
    return {
        "std_auroc": standard_vad_auroc(scores, frame_labels),
        "dc_disc": dc_disc_auroc(scores, frame_labels, num_classes),
        "dc_det_delta": dc_det_delta(scores, frame_labels, num_classes),
        "dc_sel_delta": dc_sel_delta(scores, event_spans),
    }
