# G2Rec: Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation |
| 作者 | Ruizhong Qiu, Yinglong Xia, Dongqi Fu, Hanqing Zeng, Ren Chen, Xiangjun Fan, Hong Li, Hong Yan, Hanghang Tong |
| 机构 | University of Illinois Urbana-Champaign (UIUC); Meta Monetization Recommendation Systems (MRS) |
| arXiv | https://arxiv.org/abs/2606.20554 |
| 提交日期 | 2026-06-18 |
| 领域标签 | 生成式推荐 · 图神经网络 · 语义 ID · 用户行为建模 · 工业推荐 |
| 桶类别 | WEAK |
| 综合评分 | **65 / 100** |

---

## 方法概述 (中文)

生成式推荐（Generative Recommendation）是工业推荐系统的新兴范式：通过自回归解码直接生成用户下一个感兴趣物品的 ID（Semantic ID），无需传统的检索+排序两阶段。其核心是**物品分词（Item Tokenization）**：将物品语义映射为离散 token 序列，但现有方法难以同时有效整合：

1. **用户行为协同信号**（即"喜欢 A 的用户也喜欢 B"的图结构知识）；
2. **物品语义特征**（文本/图像描述）。

**G2Rec** 提出：

1. **图感知协同用户行为建模**：构建物品共同参与图（co-engagement graph），用图神经网络在物品 embedding 中注入全局协同信号，使物品 token 同时携带内容语义+行为协同两类信息。

2. **分布式用户兴趣结构化**：用户行为历史通常包含多种分散兴趣簇（cluster），G2Rec 显式建模兴趣的多峰分布，将用户意图分解为结构化兴趣上下文注入生成式推荐模型。

3. **工业规模验证**：在 Meta MRS（Meta 广告/推荐系统）实习期间开展的工业级实验，验证在大规模系统中的有效性。

---

## 故事线 (Story Arc)

> **现状不足：** 生成式推荐的 Item Tokenization 无法同时整合图结构协同信号和物品语义；用户兴趣多峰分布导致"一个用户向量"无法完整描述其兴趣。
>
> **我们的解法：** G2Rec 用图感知 co-engagement 建模注入协同知识，再通过兴趣分布结构化捕捉用户兴趣多样性，统一在生成式推荐框架中，在 Meta 工业系统验证有效性。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 图感知物品 Tokenization + 分布式用户兴趣结构化，弥补生成式推荐的协同信号缺失 |
| 工业背景 | Meta MRS 工业合作，规模验证增加可信度 |
| vs. 先前工作 | TIGER/SASRec 等只用序列；G2Rec 显式引入图结构和兴趣多样性 |
| 局限 | 图构建的计算成本较高；兴趣 cluster 数量的超参数敏感性待探索 |

---

## 关键指标

| 实验 | 数据集/场景 | 指标 | G2Rec | Baseline |
|------|------------|------|-------|---------|
| 推荐精度 | 工业数据集 (Meta MRS) | Recall@K | 优于生成式推荐 baseline | TIGER, SASRec |
| 推荐精度 | 公开 benchmark | NDCG | 显著提升 | — |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 17 | 30 | 协同图+生成式推荐组合有新意，但各个组件单独都有相关工作 |
| Experimental SOTA delta | 9 | 15 | 工业验证背书强，但具体数值待公开 |
| Experimental quality | 10 | 15 | 公开数据集+工业数据集双验证 |
| Efficiency | 7 | 10 | 图构建可离线完成，在线推理效率合理 |
| Generalization | 4 | 5 | 生成式推荐通用框架 |
| Domain relevance | 18 | 25 | 推荐系统直接相关；电商商品推荐场景适用，但非电商特化 |
| **Total** | **65** | **100** | |

---

## 相关链接

- arXiv: https://arxiv.org/abs/2606.20554
- 机构: UIUC + Meta MRS
