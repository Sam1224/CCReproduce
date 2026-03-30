# Intern-S1-Pro (toy reproduction)

Toy reproduction skeleton for:

- **Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale** (arXiv:2603.25040)

## Idea captured

We cannot reproduce trillion-scale training. Instead, this runnable toy focuses on the *core pretraining recipe*:

- Multimodal **contrastive pretraining** (CLIP-like)
- Evaluate **zero-shot classification** by text prompts

## Quickstart

```bash
cd ccreproduce_repo/2026-03-26/intern_s1_pro
python3 -m pip install -r requirements.txt
python3 train.py --epochs 10
python3 test.py
```
