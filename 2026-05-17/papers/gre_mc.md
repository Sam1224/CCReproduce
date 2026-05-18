# GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion |
| **Authors** | Researchers from National University of Singapore |
| **ArXiv** | https://arxiv.org/abs/2605.00670 |
| **Submitted** | 1 May 2026 |
| **Venue** | arXiv preprint |
| **Code** | Not yet released |
| **Domain** | Multimodal recommendation, missing modality completion, graph neural networks |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: 多模态推荐系统在真实场景中普遍面临"模态缺失"问题——传感器故障、标注稀缺或隐私限制导致商品的视觉或文本特征不完整。现有补全方法（如简单插值、独立编码）忽略了用户-商品交互图中的丰富语义上下文，补全质量有限。  
> **我们的设计**: GRE-MC（Graph Retrieval-Enhanced Modality Completion）通过模态感知子图检索机制，从全局交互图中选取语义相关子图为缺失模态提供上下文，结合 Graph Transformer 的联合编码实现高质量补全，并通过可学习稀疏路由码本（codebook）提升鲁棒性。

### 技术细节 / Technical Details

**三大核心模块**:

1. **模态感知子图检索 (Modality-Aware Subgraph Retrieval)**:
   - 在全局用户-商品交互图中识别语义相关的子图
   - 根据不同缺失模态动态调整检索策略
   - 为补全提供丰富的上下文信息

2. **Graph Transformer 联合编码**:
   - 通过全局注意力机制联合编码查询节点和检索子图
   - 实现跨节点的上下文感知特征补全
   - 区别于独立编码的局部方法

3. **可学习稀疏路由码本 (Learnable Sparse-Routing Codebook)**:
   - 将潜在嵌入正则化为紧凑基向量
   - 提升对噪声和缺失数据的鲁棒性
   - 防止过拟合稀疏模态信息

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **检索增强补全** | 将 RAG 范式引入模态补全问题，以图检索代替静态填充，概念新颖 |
| **图上下文利用** | 利用交互图的全局结构补全局部缺失信息，比独立补全更合理 |
| **码本正则化** | 稀疏路由码本在嵌入空间提供显式正则化，有效防止退化 |
| **vs 先前工作** | 相比 BM3、FREEDOM 等多模态推荐方法，显式处理缺失模态场景，更鲁棒 |
| **实用性** | 直接针对电商实际痛点（商品图片/描述缺失），工程落地价值明确 |

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | GRE-MC | 最强基线 | 提升 |
|--------|------|--------|---------|------|
| Baby (multimodal rec) | Recall@20 | Outperforms SOTA | BM3/FREEDOM | +~3-5% |
| Sports (multimodal rec) | NDCG@20 | Outperforms SOTA | — | — |
| Clothing | Recall@20 | Outperforms SOTA | — | — |

> *注: 论文摘要表明在多个标准多模态推荐基准上一致超越SOTA，具体数值详见原文。*

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 22 | 检索增强模态补全是新颖思路，结合图结构和 codebook 设计完整 |
| Experimental SOTA delta | 15 | 10 | 多数据集一致超越，但提升幅度适中 |
| Experimental quality/ablations | 15 | 11 | 多数据集评测，各模块消融设计合理 |
| Efficiency | 10 | 6 | 子图检索增加额外计算开销，效率有待优化 |
| Generalization | 5 | 4 | 多数据集验证，泛化性较好 |
| **Domain relevance** | **25** | **17** | 电商推荐直接相关，商品模态缺失场景实际存在，但偏推荐系统侧 |
| **Total** | **100** | **70** | 多模态推荐鲁棒性方向的扎实工作 |
