# OCL — Toy Reproduction

This folder is a minimal **PyTorch** reproduction scaffold for the idea in:

**"Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems"** (arXiv:2606.04306)

The paper proposes **OCL**, a model-agnostic governance layer that sits between an LLM agent's
*proposal generation* step and the *environment execution* step, enforcing policy compliance
before any action reaches the environment.

## Architecture

```
LLM Generator
     ↓  (action proposal)
 OCL PolicyLayer
     ↓  APPROVE / BLOCK / ESCALATE
 EscalationHandler (optional human-in-the-loop)
     ↓
 Environment Execution
```

### Key Components

| Module | Description |
|--------|-------------|
| `policy.py` | PolicyLayer: rule-based + MLP classifier hybrid |
| `agent.py` | Toy LLM agent stub (buyer / seller in adversarial negotiation) |
| `environment.py` | Adversarial negotiation environment (AgenticPay-style toy) |
| `train.py` | Train the MLP policy classifier on synthetic negotiation data |
| `evaluate.py` | Evaluate safe-execution rate and valid-success rate |
| `escalation.py` | EscalationHandler: conservative fallback + logging |
| `data.py` | Synthetic action-label dataset generator |

## Quickstart

```bash
cd CCReproduce/2026-06-04/code/OCL
pip install -r requirements.txt

# Generate synthetic data and train MLP policy
python data.py --n_samples 5000 --out data/actions.jsonl
python train.py --data data/actions.jsonl --epochs 20 --ckpt checkpoints/policy.pt

# Evaluate OCL vs. no-OCL baseline in adversarial negotiation
python evaluate.py --ckpt checkpoints/policy.pt --n_episodes 200
```

## Notes

- The LLM generator is **stubbed** (rule-based heuristic + random perturbation to simulate adversarial buyer).
- To plug in a real LLM, replace `BuyerAgent.generate_action()` with an API call to your LLM.
- PolicyLayer has two modes: `rules-only` (symbolic, no training) and `hybrid` (rules + MLP).
- This is a **faithful pattern reproduction** of OCL's proposal-execution separation and policy enforcement pipeline; experimental numbers from the paper require the real AgenticPay environment.
