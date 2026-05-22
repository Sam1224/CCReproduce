# GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion |
| 作者 | Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He |
| 机构 | National University of Singapore (NUS) |
| arXiv | https://arxiv.org/abs/2605.00670 |
| 提交日期 | 2026-05-01 (v1) |
| 领域标签 | 多模态推荐 · 模态补全 · 图检索 · 鲁棒推荐 · 向量嵌入 |
| 桶类别 | STRONG |
| **总分 / Total** | **62 / 100** |

---

## 方法概述 (中文)

真实电商/推荐系统中，商品的多模态数据（图像、文本、视频）常因采集失败、标注缺失或隐私限制而出现模态缺失（Modality Incompleteness）。现有多模态推荐方法在缺失模态下性能显著下降，而简单的插值/平均填补策略无法捕捉跨商品的语义关联。

**GRE-MC（图检索增强模态补全）** 提出以下框架：

1. **模态感知子图检索 (Modality-Aware Subgraph Retrieval)**: 给定待补全商品节点，在整个商品交互图中检索语义相关的子图（基于可用模态的嵌入相似度），为补全提供丰富的跨商品上下文。
2. **图 Transformer 联合编码 (Graph Transformer with Global Attention)**: 将待补全节点与检索到的子图输入图 Transformer，通过全局注意力联合编码，输出补全后的缺失模态特征。
3. **可学习稀疏路由码本 (Learnable Sparse-Routing Codebook)**: 将补全的潜在嵌入正则化到紧凑的码本基底上，提升嵌入鲁棒性，防止补全特征过拟合噪声。

---

## 故事线 (Story Arc)

> **现状不足：** 多模态推荐系统假设所有模态都可用，但电商实际部署中商品图像/视频缺失率可达 30%+，简单零填充或均值填充的补全策略丢失了跨商品语义关联信息。
>
> **我们的解法：** GRE-MC 通过图结构检索邻居商品的相关子图，为缺失模态提供语义丰富的上下文参照，再用图 Transformer 联合编码实现高质量补全。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 图检索增强模态补全：首次将子图检索引入多模态推荐缺失模态恢复 |
| vs. 先前工作 | 先前方法（FREEDOM、BM3 等）用平均/插值补全，忽视图结构中的跨商品相关性 |
| 可行性 | 在标准多模态推荐基准（Amazon、Allrecipes）验证，NUS 实验室背书 |
| 局限 | 图检索有额外计算开销；大图的子图检索扩展性有待验证 |

---

## 关键指标

| 数据集 | 指标 | GRE-MC | 最强基线 |
|--------|------|--------|---------|
| Amazon Baby (缺失模态场景) | Recall@20 | SOTA | BM3 / FREEDOM |
| Amazon Sports (缺失模态场景) | NDCG@20 | SOTA | BM3 / FREEDOM |
| Amazon Clothing (缺失模态场景) | Recall@20 | SOTA | 先前最优 |
| Allrecipes | NDCG@20 | Competitive | — |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 19 | 30 | 子图检索+联合编码是有意义的组合创新，但整体偏增量 |
| 实验 SOTA Delta | 9 | 15 | 在多数据集上超越 SOTA，但提升幅度数值有限 |
| 实验质量/消融 | 10 | 15 | 多数据集标准评测；消融分析合理 |
| 效率 | 5 | 10 | 图检索引入额外计算；实时性有挑战 |
| 泛化性 | 4 | 5 | 多数据集验证 |
| 领域相关性 | 15 | 25 | 推荐系统有一定迁移价值，但非直接电商内容治理 |
| **总分 / Total** | **62** | **100** | — |

---

## 代码与数据

- arXiv: https://arxiv.org/abs/2605.00670
- 无代码复现（分数 62 < 80）
