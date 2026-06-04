# TrajSafe + HarmAmp Reproduction

Faithful PyTorch reproduction of:
> "Investigating and Alleviating Harm Amplification in LLM Interactions" (arXiv:2606.02423)
> Ruohao Guo, Wei Xu, Alan Ritter — Georgia Institute of Technology

## Components

```
TrajSafe/
├── harmamp/
│   ├── benchmark.py       # HarmAmp benchmark (12 risk categories, multi-turn)
│   └── categories.py      # Risk category definitions
├── trajsafe/
│   ├── monitor.py         # TrajSafe proactive monitor (tree-based RL)
│   ├── trajectory.py      # Conversation trajectory representation
│   └── intervention.py    # Intervention strategies (probe + steer)
├── training/
│   ├── tree_rl.py         # Tree-based RL training for TrajSafe
│   └── reward.py          # Reward functions
├── evaluation/
│   └── eval.py            # Benchmark evaluation
├── train.py               # Training entry point
└── evaluate.py            # Evaluation entry point
```
