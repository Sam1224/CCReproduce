# GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion |
| **Authors** | Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He |
| **Affiliation** | National University of Singapore (NUS) |
| **arXiv** | https://arxiv.org/abs/2605.00670 |
| **Submitted** | May 2026 |
| **Domain** | Multimodal Recommendation · Missing Modality · Graph Retrieval · Modality Completion |

---

## 方法概述 / Method Summary

多模态推荐系统依赖图像和文本等多模态信息提升表征质量。但真实世界数据中存在**模态缺失（modality incompleteness）**问题（传感器故障、标注稀缺、隐私约束等），会显著降低模型性能。

**GRE-MC（Graph Retrieval-Enhanced Modality Completion）** 提出：
1. **模态感知子图检索（Modality-Aware Subgraph Retrieval）：** 从整个用户-商品图中检索语义相关的子图，为缺失模态补全提供更丰富的上下文信息
2. **图 Transformer 联合编码：** 通过全局注意力机制对查询节点和检索到的子图进行联合编码，完成缺失特征的补全
3. **可学习稀疏路由码本（Learnable Sparse-Routing Codebook）：** 将隐式嵌入正则化为紧凑的基向量，提升表征鲁棒性

---

## 故事弧线 / Story Arc

> 真实电商多模态数据普遍存在模态缺失，直接降低推荐质量 → GRE-MC 从图中检索相关子图提供上下文，图 Transformer 联合编码补全缺失模态 → 在多个基准上持续超越 SOTA

---

## 创新分析 / Innovation Analysis

- 现有模态补全方法多依赖节点级特征插值，GRE-MC 引入**子图级上下文**，提供更丰富的语义信息
- 图 Transformer 全局注意力 + 子图 = 兼顾局部结构和全局语义
- 码本正则化提升了补全嵌入的鲁棒性，对噪声更不敏感

---

## 关键指标 / Key Metrics

| Dataset | Metric | GRE-MC vs SOTA |
|---------|--------|----------------|
| Amazon (Baby/Sports/Clothing) | Recall@10, NDCG@10 | Consistently outperforms |
| Modality completion benchmarks | Feature fidelity | Improved vs. interpolation |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 子图检索用于模态补全，思路新颖 |
| Experimental SOTA delta | 10 | 15 | 多基准持续提升 |
| Experimental quality | 10 | 15 | 消融分析合理 |
| Efficiency | 6 | 10 | 子图检索有额外开销 |
| Generalization | 4 | 5 | 多数据集验证 |
| Domain relevance | 18 | 25 | 多模态推荐，电商场景适用 |
| **Total** | **70** | **100** | |

