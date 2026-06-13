# OneRetrieval (Toy Reproduction)

- Paper: **OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model**
- arXiv: https://arxiv.org/abs/2606.13533

## What’s implemented

这是一个 **toy but runnable** 的 PyTorch 复现，聚焦论文的两条主线：

1) **Keyword-Aligned Encoding（KAE）的“多槽位离散码”**：把 item 的标识表示为固定长度的多 codebook token（默认 6 个槽位），每个槽位对应可解释的“语义锚点属性”（category/brand/color/style/price/material）。模型从 query 文本直接预测每个槽位的 token（多头分类），形成离散码用于检索。

2) **Editability（无需 retrain 的当天干预）**：在 toy serving 中实现了论文式的“干预 bypass”接口：
- 为每个 codebook 预留 `RES_*` reserved slots；
- 运营侧可把一个新关键词（如 `trend:viral`）绑定到某个槽位的 reserved token；
- 同时把目标 item 集合的索引码在该槽位打上同一个 reserved token；
- 推理时若 query 命中新关键词，则强制把预测码的对应槽位置为 reserved token，从而实现“新词 → 目标集合”的 **小时级生效**。

评测上提供两组指标：
- **HR@K**：常规检索命中率（query 的真实正样本是否在 top-K）。
- **IAR@K / IHR@K**：干预召回（Injected-Attribute Recall / Hit Rate），衡量新词干预是否把目标集合召回。

## Limitations (vs. the full paper)

- 原文采用 BART-base 生成序列并在真实电商检索链路做离线/在线 A/B；这里用 **非自回归的多槽位分类器** 近似 generative retrieval 的离散码生成，重点在“多槽位语义对齐 + reserved slots + serving bypass”的闭环。
- 原文的多路召回统一、四阶段 SFT、以及更复杂的 editability 指标（IAR/IHR 的工业定义）在此仅保留接口与 toy 指标形式。

## Quickstart

```bash
cd 2026-06-12/OneRetrieval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 训练并评测（包含常规 HR@K 与 editability IAR/IHR）
python train.py --out_dir runs/oneretrieval
python test.py  --ckpt_dir runs/oneretrieval
```
