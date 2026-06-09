# AdaGRPO (Toy Reproduction)

- Paper: **Adaptive Loss Balancing for Noise-Robust GRPO in Generative Recommendation**
- arXiv: https://arxiv.org/abs/2606.08480

## What’s implemented

这是一个 **toy but runnable** 的 PyTorch 复现，聚焦论文的核心机制：

1. **Generative Retrieval policy**：给定用户历史，生成一个长度为 `L=3` 的 *Semantic ID token* 序列来代表 item。
2. **Reward model（生产 ranker 的替身）**：对候选 item 打分，但对“长尾 item”存在**曝光偏置导致的局部失真**（用可控噪声模拟）。
3. **GRPO**：在每个样本上采样 `K` 个候选，按组内归一化的 advantage 做 policy gradient。
4. **AdaGRPO gating（论文核心）**：仅在同时满足
   - **policy-side difficulty**：模型对该样本不确定（GT 在候选集的 logprob 排名较差）
   - **reward discriminability**：reward model 在该样本上能把 GT 与 distractors 拉开

   时才启用 GRPO 更新；否则退化为纯 SFT（NLL）。

## Limitations (vs. the full paper)

- 真实论文基于工业级电商数据与复杂的 ranker reward；这里用合成数据与可控 bias 模拟。
- 论文中对“难度/可分性”的定义更细致；这里实现的是可运行且可解释的对齐版本。

## Quickstart

```bash
cd 2026-06-09/AdaGRPO
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) SFT 预训练 + 2) RL 微调（GRPO / AdaGRPO）
python train.py --out_dir runs/demo --mode adagrpo

# 评测 HR@10 与 hallucination（无效 Semantic ID）率
python test.py --ckpt_dir runs/demo
```

输出会包含：SFT-only、GRPO、AdaGRPO 三种模式在同一套 toy 数据上的 HR@10 / hallucination 对比。