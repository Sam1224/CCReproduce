# AgentOPSD

PyTorch reproduction scaffold for **AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning**.

This implementation covers the paper's core mechanism: token-level teacher/student log-probability gaps are aggregated into turn-level evidence, a trajectory success belief is recursively updated in log-odds space, and marginal belief revisions are used as turn-level policy weights. The included toy dataset mimics long-horizon WebShop-style agent trajectories with sparse terminal rewards and hidden pivotal turns.

## Files

- `model.py`: tiny agent policy, AgentOPSD recursive credit assignment, weighted policy objective.
- `dataset.py`: toy multi-turn trajectory dataset aligned with the training and test scripts.
- `train.py`: complete training pipeline combining imitation loss with AgentOPSD-weighted policy loss.
- `test.py`: validates pivotal-turn credit recovery and runs a training smoke test.

## Quick start

```bash
python test.py
python train.py --epochs 5
```

Full-scale reproduction requires replacing the toy dataset with ALFWorld/WebShop/Search-QA rollouts and connecting the log-probability tensors to a privileged-context teacher branch and standard-context student branch from the target LLM policy.
