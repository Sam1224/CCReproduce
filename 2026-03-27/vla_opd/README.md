# VLA-OPD (toy reproduction)

Toy reproduction skeleton for:

- **VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models ...** (arXiv:2603.26666)

## Idea captured

We implement a minimal two-stage learning setup:

1) **Offline SFT**: behavior cloning on an expert dataset.
2) **Online RL**: REINFORCE fine-tuning in a toy environment.

The environment is a small grid world; observation is a vector; action is {up,down,left,right}.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/vla_opd
python3 -m pip install -r requirements.txt

python3 train.py --stage sft --epochs 10
python3 train.py --stage rl --episodes 600
python3 test.py
```
