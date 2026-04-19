from __future__ import annotations

import math

from dataset import QAExample, build_dataset


def rule_model_predict(ex: QAExample) -> tuple[str, float]:
    # Predict answer if context contains explicit evidence; otherwise abstain.
    ctx = ex.context.lower()
    if "contains a" in ctx and ":" in ctx:
        # confidence derived from evidence presence
        for c in ["red", "blue", "green", "black", "white"]:
            if c in ctx:
                return c, 0.9
        return "<abstain>", 0.6
    return "<abstain>", 0.9


def confusion_5way(ex: QAExample, pred: str) -> str:
    # Five outcomes inspired by the paper’s A/UA × Answer/Abstain decomposition.
    if ex.is_answerable and pred != "<abstain>":
        return "AA" if pred == ex.answer else "AW"
    if ex.is_answerable and pred == "<abstain>":
        return "AABS"
    if (not ex.is_answerable) and pred == "<abstain>":
        return "UABS"
    return "UA"  # unanswerable but answered


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def abstention_mcc(dataset: list[QAExample], threshold: float) -> float:
    # Binary task: should answer? (answerable=1). Predict positive if confidence>=threshold and not abstain.
    tp = tn = fp = fn = 0
    for ex in dataset:
        pred, conf = rule_model_predict(ex)
        will_answer = (pred != "<abstain>") and (conf >= threshold)
        y = 1 if ex.is_answerable else 0
        if will_answer and y == 1:
            tp += 1
        elif will_answer and y == 0:
            fp += 1
        elif (not will_answer) and y == 0:
            tn += 1
        else:
            fn += 1
    return mcc(tp, tn, fp, fn)


def main() -> None:
    ds = build_dataset()

    counts = {"AA": 0, "AW": 0, "AABS": 0, "UABS": 0, "UA": 0}
    for ex in ds:
        pred, _ = rule_model_predict(ex)
        counts[confusion_5way(ex, pred)] += 1

    print("5-way outcomes:", counts)

    best = (-1.0, None)
    for thr in [i / 20 for i in range(1, 20)]:
        v = abstention_mcc(ds, threshold=thr)
        if v > best[0]:
            best = (v, thr)

    print(f"best_abstention_MCC={best[0]:.3f} at threshold={best[1]:.2f}")


if __name__ == "__main__":
    main()
