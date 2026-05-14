"""
ARGUS full three-stage pipeline.

Stage I  → Policy Seeding: initial model M₀ on small new-policy data
Stage II → Adversarial Label Rectification via PDU
Stage III→ Latent Knowledge Discovery: gray-area mining + hard sample synthesis

Usage:
    python pipeline.py --data data/toy_ad_governance.jsonl
"""

import argparse
import json
import os
import random
import torch
from torch.optim import AdamW

from models.ad_classifier import AdClassifier, LABEL_SPACE, LABEL_TO_IDX
from pdu.prosecutor_defender_umpire import PDUSystem
from stages.stage3_discovery import LatentKnowledgeDiscovery


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def train_classifier(model, data, n_epochs=10, lr=1e-3):
    opt = AdamW(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        random.shuffle(data)
        total_loss = 0.0
        for sample in data:
            label = sample.get("new_label", sample.get("old_label"))
            if label not in LABEL_TO_IDX:
                continue
            label_t = torch.tensor([LABEL_TO_IDX[label]])
            out = model.forward([sample["text"]], labels=label_t)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
    return total_loss / max(len(data), 1)


def evaluate_classifier(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in data:
            gold = sample.get("new_label")
            if gold not in LABEL_TO_IDX:
                continue
            pred, _ = model.predict(sample["text"])
            if pred == gold:
                correct += 1
            total += 1
    return correct / max(total, 1)


def run_argus(data_path: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_jsonl(data_path)
    train_data = [d for d in all_data if d.get("split") == "train"]
    test_data  = [d for d in all_data if d.get("split") == "test"]

    # Simulate: old-policy data has incorrect labels (= old_label)
    # New-policy data: a few samples with correct new_label (policy seeding set)
    new_policy_data = [d for d in train_data if d.get("old_label") != d.get("new_label")]
    old_policy_data = train_data

    print(f"Dataset: {len(train_data)} train / {len(test_data)} test")
    print(f"New-policy conflicting samples (seeding set): {len(new_policy_data)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE I: Policy Seeding
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  STAGE I: Policy Seeding")
    print("="*60)
    model = AdClassifier()

    # Train on all historical data (old labels — simulates stale dataset)
    stale_data = [{**d, "new_label": d["old_label"]} for d in train_data]
    loss_stale = train_classifier(model, stale_data, n_epochs=5)
    acc_before = evaluate_classifier(model, test_data)
    print(f"  Model trained on stale labels | loss={loss_stale:.4f} | test_acc={acc_before:.4f}")

    # Stage I: add new-policy seed samples
    seeding_data = [{**d} for d in new_policy_data]
    seed_loss = train_classifier(model, seeding_data, n_epochs=10, lr=2e-3)
    acc_stage1 = evaluate_classifier(model, test_data)
    print(f"  After Policy Seeding (Stage I) | loss={seed_loss:.4f} | test_acc={acc_stage1:.4f}")
    torch.save(model.state_dict(), f"{output_dir}/model_stage1.pt")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE II: Adversarial Label Rectification (PDU)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  STAGE II: Adversarial Label Rectification (PDU)")
    print("="*60)
    pdu = PDUSystem()

    ids = [d["id"] for d in train_data]
    texts = [d["text"] for d in train_data]
    hist_labels = [d["old_label"] for d in train_data]

    results = pdu.rectify(ids, texts, hist_labels)

    rectified_data = []
    gray_area_count = 0
    for r, orig in zip(results, train_data):
        rectified_data.append({**orig, "new_label": r.rectified_label})
        if r.is_gray_area:
            gray_area_count += 1
        print(
            f"  [{r.sample_id}] hist={r.historical_label:10s} → "
            f"rect={r.rectified_label:10s} | {r.umpire_reasoning[:70]}"
        )

    print(f"\n  Gray-area samples identified: {gray_area_count}/{len(train_data)}")

    # Fine-tune PDU on rectified labels
    pdu_loss = pdu.train_on_rectified(
        texts,
        [r.rectified_label for r in results],
        n_steps=10,
    )
    print(f"  PDU fine-tuned | avg_loss={pdu_loss:.4f}")

    # Retrain classifier on rectified labels
    rect_loss = train_classifier(model, rectified_data, n_epochs=10, lr=5e-4)
    acc_stage2 = evaluate_classifier(model, test_data)
    print(f"  After Label Rectification (Stage II) | loss={rect_loss:.4f} | test_acc={acc_stage2:.4f}")
    torch.save(model.state_dict(), f"{output_dir}/model_stage2.pt")

    # Save rectification results
    with open(f"{output_dir}/rectification_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({
                "id": r.sample_id,
                "historical": r.historical_label,
                "rectified": r.rectified_label,
                "prosecutor_score": r.prosecutor_score,
                "defender_score": r.defender_score,
                "is_gray_area": r.is_gray_area,
            }) + "\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE III: Latent Knowledge Discovery
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  STAGE III: Latent Knowledge Discovery")
    print("="*60)
    discovery = LatentKnowledgeDiscovery(model=model)

    gray_samples = [
        (r, orig) for r, orig in zip(results, train_data) if r.is_gray_area
    ]
    if gray_samples:
        hard_samples = discovery.mine_hard_samples(
            [(r.ad_text, r.rectified_label) for r, _ in gray_samples]
        )
        print(f"  Mined {len(hard_samples)} hard adversarial samples from {len(gray_samples)} gray-area cases")
        augmented_data = rectified_data + hard_samples

        # Final training on augmented data
        final_loss = train_classifier(model, augmented_data, n_epochs=10, lr=3e-4)
        acc_stage3 = evaluate_classifier(model, test_data)
        print(f"  After Latent Knowledge Discovery (Stage III) | loss={final_loss:.4f} | test_acc={acc_stage3:.4f}")
    else:
        acc_stage3 = acc_stage2
        print("  No gray-area samples found. Stage III skipped.")

    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  ARGUS PIPELINE SUMMARY")
    print("="*60)
    print(f"  Baseline (stale labels):       {acc_before:.4f}")
    print(f"  After Stage I (Policy Seed):   {acc_stage1:.4f}")
    print(f"  After Stage II (PDU Rectify):  {acc_stage2:.4f}")
    print(f"  After Stage III (Discovery):   {acc_stage3:.4f}")
    print(f"  Improvement over baseline:     {acc_stage3 - acc_before:+.4f}")
    print(f"\n  Outputs saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_ad_governance.jsonl")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    run_argus(args.data, args.output_dir)
