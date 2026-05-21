# GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion |
| **作者** | Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He |
| **机构** | National University of Singapore |
| **arXiv ID** | [2605.00670](https://arxiv.org/abs/2605.00670) |
| **提交日期** | May 1, 2026 |
| **代码** | Not yet public |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** Multimodal recommendation systems rely on visual and textual item features, but real-world datasets suffer from **modality incompleteness** — missing images due to vendor non-compliance, broken links, or privacy regulations. Most existing methods either ignore missing modalities or fill them with mean embeddings, which degrades performance significantly.

GRE-MC proposes a **graph retrieval-enhanced modality completion** framework. Given an item with missing modality, it executes a **modality-aware subgraph retrieval**: it identifies semantically similar neighbors in the item-user interaction graph (considering both available modalities and collaborative signals), retrieves a semantically relevant subgraph, and uses this subgraph as context to complete the missing modality via a learned generator. The completed modality representation is then fed into a standard graph-based collaborative filtering backbone (BM3/LightGCN variants). A consistency regularization term enforces that completed features align with the distribution of observed features.

**ZH:** GRE-MC 解决多模态推荐中的模态缺失问题。对于缺失模态的商品，方法在用户-商品交互图上执行**模态感知子图检索**，找到语义相似邻居并以子图为上下文通过学习生成器补全缺失模态。一致性正则化确保补全特征与观测特征分布对齐，补全后接入标准图协同过滤骨干网络。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 多模态推荐假设所有商品均有完整的图文特征，实际电商场景中因供应商未上传图片、链接失效等原因大量商品存在模态缺失，简单均值填充会引入大量噪声。
>
> **我们的解决方案:** 利用图结构中的语义邻居作为"示例"来引导缺失模态的生成，比均值填充或随机噪声补全更准确地恢复商品特征，提升了推荐鲁棒性。

---

## 创新性分析 / Innovation Analysis

1. **检索增强补全：** 将 RAG 思路引入模态补全，以图结构邻居为上下文，比单纯生成更可控。
2. **模态感知子图检索：** 同时考虑可用模态相似性和协同信号，比单纯基于内容的检索更准确。
3. **即插即用：** 补全模块可与任意图推荐骨干网络结合，不绑定特定模型架构。
4. **vs. 先前工作：** LATTICE、BM3 等不处理模态缺失；MMSSL 用固定均值填充；GRE-MC 提供了动态、上下文感知的解决方案。

---

## 关键指标 / Key Metrics

| Dataset | Metric | GRE-MC | Best Baseline | Δ |
|---------|--------|--------|---------------|---|
| Baby (30% missing) | Recall@20 | ~0.0712 | ~0.0641 | +11.1% |
| Sports (30% missing) | Recall@20 | ~0.0598 | ~0.0541 | +10.5% |
| Clothing (30% missing) | Recall@20 | ~0.0423 | ~0.0389 | +8.7% |

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 18/30 | RAG 思路迁移到模态补全，有新意但基础技术成熟 |
| Experimental SOTA delta | 10/15 | ~10% Recall 提升，适中 |
| Experimental quality | 12/15 | 三数据集，多缺失率测试 |
| Efficiency | 6/10 | 子图检索有额外计算代价 |
| Generalization | 3/5 | 跨缺失率泛化好，跨领域未验证 |
| Domain relevance | 21/25 | 直接对应电商多模态商品推荐，缺失模态是真实痛点 |
| **Total** | **70/100** | |
