## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | ELVA: Exploring Ranking-Driven Universal Multimodal Retrieval |
| **Short Name** | ELVA |
| **arXiv ID** | 2606.20280 |
| **Link** | https://arxiv.org/abs/2606.20280 |
| **PDF** | https://arxiv.org/pdf/2606.20280 |
| **Authors** | (see arxiv page) |
| **Submitted** | 2026-06-18 |
| **Venue** | arXiv preprint |
| **Tags** | 多模态检索, 强化学习, 对比学习, 通用检索, 电商搜索 |

---

## 故事弧 / Story Arc

**现有问题**: MLLM 驱动的通用多模态检索（Universal Multimodal Retrieval, UMR）依赖对比学习，把所有 negative 样本视为二元分类（正/负），忽略不同 negative 携带的粒度信息差异，导致模型产生**粒度盲区（grain blindness）**——对复杂查询中的细粒度信息不敏感。

**解决方案**: ELVA 用基于规则的强化学习（rule-based RL）对 negative 按与 positive 的相似度排序，让模型从每个 negative 中学习不同粒度的区分信息，缓解粒度盲区。

---

## 方法概述 / Method Overview

- **Grain Blindness 的根本原因**: 对比学习的 InfoNCE loss 把所有 negative 等价对待，导致模型无法区分"硬负样本（hard negative，与 positive 语义相近）"和"软负样本（easy negative，完全无关）"，从而错过细粒度区分训练信号。

- **ELVA 框架**:
  1. 对每个训练 anchor，将 batch 内 negatives 按与 positive 的余弦相似度排序
  2. 定义规则奖励（rule reward）：根据排序位置对 negative 给予不同权重/损失系数
  3. 用 RL 风格优化（类似 GRPO）迭代更新检索模型
  4. 模型同时从粗粒度（区分无关样本）和细粒度（区分相似样本）的信号中学习

---

## 关键指标 / Key Metrics

详见论文。在标准 UMR 基准（如 M-BEIR）上相较对比学习 baseline 有提升。

---

## 评分 / Scoring

| 维度 | 分值 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 21 | 30 | RL 排序 negative 的思路新颖，但 RL in retrieval 已有先例 |
| Experimental SOTA delta | 10 | 15 | 标准 UMR 基准提升，幅度待确认 |
| Experimental quality | 10 | 15 | 需看消融实验完整性 |
| Efficiency | 7 | 10 | RL 训练额外成本，推理期与标准检索相同 |
| Generalization | 3 | 5 | UMR 框架通用，但未见跨模态泛化深度分析 |
| Domain relevance | 17 | 25 | 多模态检索可用于电商商品搜索/内容检索，但未直接针对电商 |
| **Total** | **68** | **100** | |
