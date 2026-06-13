# Influcoder (Toy Reproduction)

- Paper: **Influcoder: Distilling Decoders’ Gradient Influence Rankings into an Encoder for Data Attribution**
- arXiv: https://arxiv.org/abs/2606.13668

## What’s implemented

这是一个 **toy but runnable** 的复现，聚焦论文主链路：

1) **Stage-1：decoder 侧 influence ranking**
- 训练一个小型“decoder-style”文本分类模型（用 LoRA 适配最后一层线性分类器）。
- 对每条样本计算 **LoRA 参数梯度**（近似论文中的“对 response-only 的 LoRA gradients”），并对梯度向量做 **CountSketch 压缩**。
- 用压缩后的梯度向量做 cosine similarity，作为 query→pool 的 influence 分数（ground-truth）。

2) **Stage-2：distill 到 encoder embedding**
- 训练一个轻量 encoder，把文本映射为 influence embedding。
- 训练目标：让 encoder embedding 的 cosine similarity 拟合 Stage-1 的 ground-truth influence 分数。

3) **评测**
- **Spearman rank correlation**：每个 query 的“pool 排名”与 ground-truth 的一致性。
- **AUPRC（toy toxicity filtering）**：把 influence 分数当作“找出有毒样本”的排序分数。

## Limitations (vs. the full paper)

- 原文以 LLM decoder + LoRA gradients 为基础，并在更真实的数据选择/过滤任务上评估；这里用可控的合成数据与小模型，保留接口与关键步骤。
- 原文的损失包含 Pearson ranking + KL 等组合，这里用 **pairwise cosine regression**（MSE）做最小可跑闭环。

## Quickstart

```bash
cd 2026-06-13/Influcoder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 训练并评测
python train.py --out_dir runs/influcoder
python test.py  --ckpt_dir runs/influcoder
```
