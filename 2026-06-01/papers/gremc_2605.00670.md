# Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion (GRE-MC)

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion |
| **Authors** | Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He |
| **Affiliation** | National University of Singapore (NUS) |
| **Venue** | arXiv preprint |
| **arXiv** | https://arxiv.org/abs/2605.00670 |
| **Submitted** | 2026-05-01 |

---

## 方法概述 / Method Summary

真实世界多模态推荐数据集中，商品的视觉或文本模态往往不完整（传感器故障、标注稀缺、隐私限制），导致模型性能大幅下降。现有模态补全方法仅利用节点自身或邻居信息进行缺失模态重建，忽略了图中语义相关的更广泛上下文。

本文提出 **GRE-MC（Graph Retrieval-Enhanced Modality Completion）**：

1. **模态感知子图检索（Modality-Aware Subgraph Retrieval）**：给定有缺失模态的查询节点，从全图中检索语义相关子图（而非仅用一阶邻居），提供更丰富的上下文。
2. **联合编码图变换器（Joint-Encoding Graph Transformer）**：通过全局注意力机制联合编码查询节点与检索到的子图，完成缺失模态特征的重建。
3. **可学习稀疏路由码本（Learnable Sparse-Routing Codebook）**：对潜在 embedding 进行正则化，防止重建过拟合，提升鲁棒性。

**故事弧线：** 现有模态补全方法视野局限于局部邻居，丢失了图中大量语义上下文 → GRE-MC 以子图检索扩大感受野，图变换器联合编码，在多模态推荐基准上持续超越 SOTA。

---

## 评分 / Score

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | 子图检索用于模态补全有新意，但整体范式相对渐进 |
| Experimental SOTA Delta | 11 | 15 | 多基准持续 SOTA，具体数值未在摘要中披露 |
| Experimental Quality / Ablations | 12 | 15 | NUS 团队，多基准实验 |
| Efficiency | 7 | 10 | 图变换器计算适中 |
| Generalization | 4 | 5 | 多基准验证 |
| Domain Relevance (ecom + governance) | 16 | 25 | 多模态推荐（可用于电商），但非直接电商场景或内容治理 |
| **Total** | **70** | **100** | |
