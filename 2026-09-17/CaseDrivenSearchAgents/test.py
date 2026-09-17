from __future__ import annotations

import os

import torch

from data import ToyRelevanceDataset, build_cfg_from_ckpt, evaluate
from model import AllInOneRelevanceModel, ModelConfig
from train import AGENT_CKPT, BASELINE_CKPT


def load_payload(path: str, device: torch.device):
    payload = torch.load(path, weights_only=False, map_location=device)
    cfg = build_cfg_from_ckpt(payload)
    model = AllInOneRelevanceModel(ModelConfig(**payload["model_cfg"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload, cfg, model


def main() -> None:
    if not (os.path.exists(BASELINE_CKPT) and os.path.exists(AGENT_CKPT)):
        raise SystemExit("missing checkpoints; run `python train.py` first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_payload, cfg, baseline = load_payload(BASELINE_CKPT, device)
    agent_payload, _, agent_model = load_payload(AGENT_CKPT, device)

    test_ds = ToyRelevanceDataset(1500, cfg, seed=1)
    base_metrics = evaluate(baseline, test_ds)
    agent_metrics = evaluate(agent_model, test_ds)

    print("=== Case-driven multi-agent relevance ===")
    print("baseline:", base_metrics)
    print("multi-agent:", agent_metrics)
    print("memory updates:", len(agent_payload.get("memory", {}).get("standard_updates", [])))
    print("expectation: multi-agent loop improves win_rate / relevant precision-recall on bad-case-heavy evaluation")


if __name__ == "__main__":
    main()
