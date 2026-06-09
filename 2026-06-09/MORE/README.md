# MORE (Toy Reproduction)

- Paper: **One Model, Multiple Goals: Adaptive Multi-Objective Learning for E-commerce Dialogue Systems**
- arXiv: https://arxiv.org/abs/2606.09293

## What’s implemented

这是一个 **toy but runnable** 的 PyTorch 复现，用合成的“电商对话”数据去对齐论文的两点核心思想：

1. **多目标学习的真实矛盾**：
   - 目标 A（Reasoning / Correctness）：必须基于用户 profile（额度/是否可分期等）给出正确答案。
   - 目标 B（Generation / Naturalness）：回答要更自然（更接近参考表达，而不是僵硬模板）。

2. **MORE 的两条关键做法（toy 对齐版）**：
   - **把 reasoning 当成“约束/脚手架”**：当回答不满足正确性时，优先把优化压力放在 correctness 上；推理正确后再重点优化生成质量。
   - **自适应 multi-reward**：在 correctness 达标后，对“BLEU（贴近参考）/长度/多样性”等生成奖励做动态 reweight（toy 里用 moving-average+variance 的简化版本）。

## Limitations (vs. the full paper)

- 原文是工业级线上系统（多 reward、GRPO、reasoning refinement 等）；这里用 REINFORCE + 合成数据做 runnable 对齐。
- 复现目标是“把机制跑通并可对比”，不是复刻线上绝对数值。

## Quickstart

```bash
cd 2026-06-09/MORE
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SFT 预训练 + RL 微调（baseline: fixed-sum vs MORE）
python train.py --out_dir runs/demo --mode more

python test.py --ckpt_dir runs/demo
```

评测输出包含：correctness accuracy、BLEU-2、以及简单的多样性指标（unique trigrams）。