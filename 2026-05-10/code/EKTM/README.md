# EKTM: Code Reproduction

**Paper:** Effective Knowledge Transfer for Multi-Task Recommendation Models  
**arXiv:** https://arxiv.org/abs/2605.05730  
**Score:** 82/100

## Architecture

```
Input Features (user + item + context)
        ↓
  Shared Bottom MLP
        ↓
 ┌──────┼──────┐
Task0  Task1  ...  (per-task Expert Networks)
 │      │
 └──────┴──── Router Module ─────────┐
              (cross-task attention) │
                                     ↓ router_out
               ┌─────────────────────┴──────┐
           Transmitter_0          Transmitter_1    (per-task)
               │                        │
           Enhanced_0               Enhanced_1     (residual gate)
               │                        │
           Output_0                 Output_1       (binary CVR logit)
```

## Files

| File | Description |
|---|---|
| `model.py` | Full EKTM architecture: RouterModule, TransmitterModule, EnhancedModule, EKTMTaskTower, EKTM |
| `train.py` | Training loop with multi-task BCE loss, AdamW, cosine LR, AUC/GAUC eval |
| `eval.py` | Evaluation with AUC, GAUC, AP metrics |

## Quick Start

```bash
pip install torch scikit-learn numpy

# Train (uses toy data with interface-aligned to production)
python train.py

# Evaluate saved checkpoint
python eval.py
```

## Key Design Decisions

1. **Two-pass forward**: Pass 1 computes initial task representations; Router aggregates; Pass 2 applies Transmitter+Enhanced with router knowledge. This mirrors the paper's description of Router receiving all task signals before transmitting.

2. **Per-task Transmitter gating**: Each transmitter uses a learned gate (sigmoid) to control how much router knowledge enters the task, preventing hard injection of irrelevant cross-task signals.

3. **Enhanced module beta**: A learned scalar `beta` (initialized to 0, activated via sigmoid → range [0,1]) controls injection strength. Starting at 0 ensures stable early training.

4. **GAUC metric**: Group AUC (per-user AUC averaged by impression count) is the standard industrial recommendation metric — implemented in `train.py::compute_gauc`.

## Correspondence to Paper

| Paper Component | Code Location |
|---|---|
| Router module (Section 3.2) | `model.py::RouterModule` |
| Transmitter module (Section 3.3) | `model.py::TransmitterModule` |
| Enhanced module (Section 3.4) | `model.py::EnhancedModule` |
| Multi-task CVR loss | `train.py::train_ektm` |
| GAUC evaluation | `train.py::compute_gauc` |
