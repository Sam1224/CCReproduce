from __future__ import annotations

import argparse
import math

import torch

from dataset import SyntheticUser, build_synthetic_users
from metrics import interest_groundedness, interest_specificity
from model import InterestExtractor


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearmanr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 0 else float("nan")


def collate_one(user: SyntheticUser, *, leaf_index: dict[str, int], max_len: int):
    leaf_ids = []
    weights = []
    for interaction in user.history[:max_len]:
        leaf_ids.append(leaf_index[interaction.category])
        weights.append(float(interaction.weight))

    mask = [1.0] * len(leaf_ids)
    while len(leaf_ids) < max_len:
        leaf_ids.append(0)
        weights.append(0.0)
        mask.append(0.0)

    return (
        torch.tensor(leaf_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(weights, dtype=torch.float32).unsqueeze(0),
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu")
    leaf_vocab = payload["leaf_vocab"]
    interest_vocab = payload["interest_vocab"]
    max_len = int(payload.get("max_len") or 64)

    leaf_index = {leaf: idx for idx, leaf in enumerate(leaf_vocab)}

    model = InterestExtractor(num_leaf_categories=len(leaf_vocab), num_interests=len(interest_vocab))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    users = build_synthetic_users(num_users=80, seed=1)

    f1_list: list[float] = []
    is_list: list[float] = []
    survey_list: list[float] = []

    for user in users:
        leaf_ids, weights, mask = collate_one(user, leaf_index=leaf_index, max_len=max_len)
        with torch.no_grad():
            logits = model(leaf_ids, weights, mask).logits.squeeze(0)
        top_indices = logits.topk(k=min(args.topk, logits.numel())).indices.tolist()
        predicted = [interest_vocab[int(i)] for i in top_indices]

        ig = interest_groundedness(predicted=predicted, ground_truth=user.gt_interests)
        is_score = interest_specificity(predicted=predicted)

        f1_list.append(float(ig["f1"]))
        is_list.append(float(is_score))
        survey_list.append(float(user.survey_score))

    combined = [0.7 * f + 0.3 * s for f, s in zip(f1_list, is_list)]

    print("mean_ig_f1", sum(f1_list) / len(f1_list))
    print("mean_is", sum(is_list) / len(is_list))
    print("spearman_rho(combined, survey)", spearmanr(combined, survey_list))


if __name__ == "__main__":
    main()
