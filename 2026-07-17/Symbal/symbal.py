import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


OBJECT_VOCAB = [
    "bus", "table", "cake", "person", "sign", "dog", "cat", "car", "chair", "handbag",
    "white tablecloth", "text", "pacemaker", "cardiomegaly", "chest tube", "edema",
]


class HashTextEncoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, texts):
        rows = []
        for text in texts:
            vec = torch.zeros(self.dim)
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            for token in tokens:
                vec[hash(token) % self.dim] += 1.0
            rows.append(vec)
        return F.normalize(torch.stack(rows), dim=-1)


class LinearImageEncoder(torch.nn.Module):
    def __init__(self, input_dim, dim=128):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, dim)

    def forward(self, image_features):
        return F.normalize(self.proj(image_features.float()), dim=-1)


def split_facts(caption):
    parts = re.split(r"(?<=[.!?。！？])\s+|;", caption.strip())
    return [part.strip() for part in parts if part.strip()]


def spherical_kmeans(x, k, iters=30):
    x = F.normalize(x, dim=-1)
    centers = x[torch.linspace(0, len(x) - 1, k).long()].clone()
    for _ in range(iters):
        labels = torch.matmul(x, centers.T).argmax(dim=1)
        new_centers = []
        for idx in range(k):
            members = x[labels == idx]
            new_centers.append(members.mean(dim=0) if len(members) else centers[idx])
        centers = F.normalize(torch.stack(new_centers), dim=-1)
    return labels, centers


def choose_k(x, max_k=8):
    if len(x) <= 2:
        return 1
    best_k, best_score = 2, -1.0
    for k in range(2, min(max_k, len(x) - 1) + 1):
        labels, centers = spherical_kmeans(x, k, iters=10)
        sims = torch.matmul(x, centers.T)
        own = sims[torch.arange(len(x)), labels]
        masked = sims.clone()
        masked[torch.arange(len(x)), labels] = -2
        nearest_other = masked.max(dim=1).values
        score = ((own - nearest_other) / (torch.maximum(own.abs(), nearest_other.abs()) + 1e-6)).mean().item()
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def summarize_text(facts):
    phrase_counts = Counter()
    lowered = " ".join(facts).lower()
    for phrase in sorted(OBJECT_VOCAB, key=len, reverse=True):
        if phrase in lowered:
            phrase_counts[phrase] += lowered.count(phrase) * max(1, len(phrase.split()))
    if phrase_counts:
        return phrase_counts.most_common(1)[0][0]
    words = [w for w in re.findall(r"[a-z][a-z0-9-]+", lowered) if len(w) > 3]
    return Counter(words).most_common(1)[0][0] if words else "unknown textual fact"


def summarize_visual(indices, image_tags):
    counts = Counter()
    for idx in indices:
        counts.update(image_tags[idx])
    return counts.most_common(1)[0][0] if counts else "unknown visual feature"


def detect_systematic_misalignment(records, image_encoder=None, text_encoder=None, top_k_text=3):
    text_encoder = text_encoder or HashTextEncoder()
    image_dim = len(records[0]["image_features"])
    image_encoder = image_encoder or LinearImageEncoder(image_dim)

    facts, fact_to_record = [], []
    for ridx, record in enumerate(records):
        for fact in split_facts(record["caption"]):
            facts.append(fact)
            fact_to_record.append(ridx)

    text_emb = text_encoder(facts)
    image_tensor = torch.tensor([r["image_features"] for r in records], dtype=torch.float32)
    image_emb = image_encoder(image_tensor)

    k_text = choose_k(text_emb)
    text_labels, _ = spherical_kmeans(text_emb, k_text)
    clusters = defaultdict(list)
    for idx, label in enumerate(text_labels.tolist()):
        clusters[label].append(idx)

    text_cluster_scores = []
    for label, indices in clusters.items():
        alignments = []
        for fact_idx in indices:
            ridx = fact_to_record[fact_idx]
            alignments.append(torch.dot(text_emb[fact_idx], image_emb[ridx]).item())
        text_cluster_scores.append((sum(alignments) / len(alignments), label, indices))
    text_cluster_scores.sort(key=lambda item: item[0])

    predictions = []
    for text_score, label, fact_indices in text_cluster_scores[:top_k_text]:
        text_cluster_facts = [facts[i] for i in fact_indices]
        erroneous_fact = summarize_text(text_cluster_facts)
        candidate_records = sorted(set(fact_to_record[i] for i in fact_indices))
        if len(candidate_records) < 2:
            candidate_records = list(range(len(records)))
        candidate_image_emb = image_emb[candidate_records]
        k_image = choose_k(candidate_image_emb)
        image_labels, _ = spherical_kmeans(candidate_image_emb, k_image)
        image_cluster_scores = []
        for image_label in sorted(set(image_labels.tolist())):
            local_indices = [i for i, value in enumerate(image_labels.tolist()) if value == image_label]
            record_indices = [candidate_records[i] for i in local_indices]
            alignments = []
            for ridx in record_indices:
                related_fact_emb = text_encoder([erroneous_fact])[0]
                alignments.append(torch.dot(related_fact_emb, image_emb[ridx]).item())
            image_cluster_scores.append((sum(alignments) / len(alignments), record_indices))
        image_cluster_scores.sort(key=lambda item: item[0])
        visual_feature = summarize_visual(image_cluster_scores[0][1], [r.get("image_tags", []) for r in records])
        predictions.append({
            "textual_error": erroneous_fact,
            "visual_feature": visual_feature,
            "text_cluster_alignment": text_score,
            "supporting_facts": text_cluster_facts[:8],
            "supporting_record_ids": image_cluster_scores[0][1][:20],
        })
    return predictions


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="predictions.json")
    args = parser.parse_args()
    records = load_records(args.data)
    predictions = detect_systematic_misalignment(records, top_k_text=args.top_k)
    Path(args.output).write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
