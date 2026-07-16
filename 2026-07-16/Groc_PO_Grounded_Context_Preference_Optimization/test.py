import argparse

import torch

from data import STAGES, make_loaders
from model import GroundedContextEncoder, GrocPOConfig
from train import evaluate


def hallucination_risk(model, loader, device):
    model.eval()
    risky = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            wrong_preferred = ((out["margin"] > 0).float() != batch["label"]).sum().item()
            risky += wrong_preferred
            total += batch["label"].numel()
    return risky / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/groc_po.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = make_loaders(args.batch_size)
    state = torch.load(args.checkpoint, map_location=device)
    model = GroundedContextEncoder(GrocPOConfig(**state.get("cfg", {}))).to(device)
    model.load_state_dict(state["model"])

    metrics = evaluate(model, test_loader, device)
    risk = hallucination_risk(model, test_loader, device)
    print("Groc-PO staged preference audit")
    print(f"overall_preference_accuracy={metrics['pref_acc']:.4f}")
    for name, sid in STAGES.items():
        print(f"stage_{name}_accuracy={metrics['stage_acc'][sid]:.4f}")
    print(f"hallucination_risk_rate={risk:.4f}")


if __name__ == "__main__":
    main()
