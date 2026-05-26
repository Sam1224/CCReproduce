# GRMCRec: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion |
| **arXiv** | [2605.00670](https://arxiv.org/abs/2605.00670) |
| **Authors** | Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He |
| **Affiliations** | National University of Singapore |
| **Date** | 2026-05-01 |
| **Bucket** | STRONG |
| **Total** | **69 / 100** |

---

## 故事弧 / Story Arc

> **问题:** 真实世界的多模态推荐系统数据集经常出现模态缺失（传感器故障、标注稀缺、隐私约束），严重影响模型性能和可靠性。现有模态补全方法多依赖静态补全，忽视了物品交互图中丰富的邻居语义信息。
>
> **方案:** GRMCR（Graph Retrieval-Enhanced Modality Completion and Recommendation）：(1) 模态感知子图检索机制，从全局交互图中选取语义相关邻居子图；(2) 图Transformer联合编码查询节点与检索子图通过全局注意力完成缺失模态特征；(3) 可学习稀疏路由码本（Codebook）正则化潜在嵌入为紧凑基向量，提升鲁棒性。
>
> **差异:** 将检索增强生成（RAG）思路引入模态补全领域，不同于现有基于生成模型的静态补全方法，动态利用交互图结构作为上下文信息源。

---

## 方法概述 / Method Summary

**三阶段框架:**

```
Incomplete Multimodal Item Graph
         ↓
1. Modality-Aware Subgraph Retrieval
   - 选取具有相似语义的邻居子图
   - 跨模态相似度加权检索
         ↓
2. Graph Transformer Completion
   - 查询节点 + 检索子图 → 全局注意力
   - 重建缺失模态特征向量
         ↓
3. Sparse-Routing Codebook Regularization
   - 将补全后嵌入映射到紧凑码本
   - 防止过拟合，提升泛化鲁棒性
         ↓
Recommendation Score
```

**核心模块:**

- **模态感知检索:** 基于视觉/文本可用模态计算跨模态相似度，检索K个最近邻子图
- **图Transformer:** 在检索子图上运行全局多头注意力，聚合邻居信息为缺失模态生成特征
- **稀疏路由码本:** $z_{complete} = \sum_k r_k \cdot c_k$，其中 $r_k$ 是稀疏路由权重，$c_k$ 是码本条目

---

## 创新性分析 / Innovation

1. **RAG思路迁移:** 将检索增强的核心思想创造性地移植到模态补全，是领域内的新颖探索
2. **图结构利用:** 充分利用用户-物品交互图的协作过滤信号，比纯视觉/文本补全更全面
3. **码本正则化:** 稀疏路由码本保持补全特征的紧凑性，是不同于VAE/GAN的有效正则化手段
4. **可行性:** NUS团队具有图神经网络和推荐系统的扎实背景

---

## 关键指标 / Key Metrics

| Dataset | Modality Missing Rate | Metric | GRMCR | Best Baseline |
|---------|----------------------|--------|-------|---------------|
| Amazon Baby/Sports/Clothing | 20%-80% | Recall@20 | +↑ | FREEDOM/SLMRec |
| Amazon Beauty | 50% missing | NDCG@20 | +↑ | — |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | RAG迁移到模态补全是新颖思路；图Transformer组合合理 |
| Experimental SOTA delta | 10 | 15 | 标准Amazon数据集上改进；缺失率不同设置下均有提升 |
| Experimental quality / ablations | 11 | 15 | 不同缺失率、不同数据集消融完整 |
| Efficiency | 5 | 10 | 子图检索增加训练开销；实际部署效率待评估 |
| Generalization | 3 | 5 | Amazon数据集验证，需要更多电商场景验证 |
| Domain relevance (ecom + governance) | 20 | 25 | 多模态推荐是电商核心能力；模态缺失是真实痛点 |
| **Total** | **69** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **电商推荐核心:** 解决商品图片缺失/文字描述缺失导致的推荐降级问题
- **新商品冷启动:** 新上架商品常缺少完整多模态信息，模态补全帮助提升推荐效果
- **达人选品推荐:** 多模态商品理解支撑达人/KOL选品和商品匹配推荐
