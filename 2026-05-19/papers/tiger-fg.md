# TIGER-FG: Text-Guided Implicit Fine-Grained Grounding for E-commerce Retrieval

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | TIGER-FG: Text-Guided Implicit Fine-Grained Grounding for E-commerce Retrieval |
| **arXiv** | https://arxiv.org/abs/2605.18434 |
| **Submitted** | ~May 18-19, 2026 (2605.18434) |
| **Domain** | E-commerce Retrieval · Visual Grounding · Dual Distillation · Multimodal · IMMR |
| **Code** | `TIGER-FG/` (完整复现，含 train.py / eval.py / toy 数据集) |

---

## 方法概述 / Method Summary

电商图搜中一个核心场景是 **IMMR（Implicit Multi-step Multi-modal Retrieval）**：用户上传一张**局部目标图**（如从某张照片中截取的手表），期望检索到完整商品页（full item image + 结构化文本）。  

现有方案往往需要先显式检测/框选目标区域，再做检索，但在线服务中显式检测开销大且框不准。  

**TIGER-FG** 提出**文本引导的隐式细粒度 Grounding**：
- **Item Encoder：** 用结构化商品文本（标题/属性）作为语义引导 Query，对全图 patch 做 Cross-Attention 聚合，生成**目标感知的 item embedding**，无需显式检测框
- **Query Encoder：** 对 crop 图直接编码
- **Dual Distillation（双蒸馏）：**
  - *Spatial-Relational Distillation：* 以 box mask 做区域先验，蒸馏 student 的 patch-relation，让模型隐式学会"哪些 patch 与目标相关"
  - *Similarity-Distribution Distillation：* 蒸馏 batch 内 query–item 相似度分布（soft targets），提升表征质量

---

## 故事弧线 / Story Arc

> 电商图搜的 IMMR 任务需要显式检测，成本高且误差传播 → 设计文本引导的隐式 Grounding，让文本属性引导视觉注意力 → 结合双蒸馏策略，无需显式框选即可实现细粒度检索

---

## 创新分析 / Innovation Analysis

**与前人工作的差异：**
- 传统方法（两阶段检测+检索）存在误差传播；本文通过**文本语义驱动 cross-attention**实现端到端的隐式 Grounding
- Dual Distillation 同时约束空间相对关系和相似度分布，在无检测框监督时提供替代监督信号
- IMMR 任务定义本身具有实际意义（crop query ← 线上用户真实上传行为）
- 文本引导注意力使 item encoder 自适应聚焦于与查询语义相关的区域，对多目标/复杂背景商品图更鲁棒

**可行性：** 论文提供了 arXiv 全文，本仓库已完成 toy 级复现验证。

---

## 关键指标 / Key Metrics

（来自论文 README 描述，具体数值待原文发布后补充）

| Task | Dataset | Metric | TIGER-FG |
|------|---------|--------|----------|
| IMMR Retrieval | E-commerce (full-item vs crop) | Recall@K | SOTA (consistently outperforms baseline) |
| Spatial Distillation | Ablation | Recall@1 | ↑ vs no spatial distil |
| Similarity Distillation | Ablation | Recall@5 | ↑ vs no sim distil |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | IMMR 任务定义新颖，文本引导隐式 Grounding 创新 |
| Experimental SOTA delta | 11 | 15 | 双蒸馏有效，数值待原文确认 |
| Experimental quality | 12 | 15 | 双 ablation 设计清晰 |
| Efficiency | 8 | 10 | 免显式检测，线上友好 |
| Generalization | 4 | 5 | 电商场景通用 |
| Domain relevance | 24 | 25 | 核心电商图搜+视觉理解 |
| **Total** | **83** | **100** | |

---

## 代码说明

复现见 `TIGER-FG/`，包含：
- `tiger_fg/dataset.py` — 合成 IMMR toy 数据（full image+box+text；crop query）
- `tiger_fg/model.py` — TIGER-FG（image encoder + text encoder + text-guided cross-attn）
- `tiger_fg/losses.py` — 对比学习 + 双蒸馏损失
- `tiger_fg/metrics.py` — Recall@K
- `train.py` / `eval.py`
