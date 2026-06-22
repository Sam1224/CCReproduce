## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation |
| **Short Name** | G2Rec |
| **arXiv ID** | 2606.20554 |
| **Link** | https://arxiv.org/abs/2606.20554 |
| **PDF** | https://arxiv.org/pdf/2606.20554 |
| **Authors** | Ruizhong Qiu, Yinglong Xia, Dongqi Fu, Hanqing Zeng, Ren Chen, Xiangjun Fan, Hong Li, Hong Yan, Hanghang Tong |
| **Affiliations** | University of Illinois Urbana-Champaign; Meta |
| **Submitted** | 2026-06-18 |
| **Venue** | arXiv preprint |
| **Tags** | 生成式推荐, 图神经网络, 语义分词, 工业部署, 电商推荐, LLM |

---

## 故事弧 / Story Arc

**现有问题**: 生成式序列推荐的核心是 item tokenization——把物品映射为离散 token 序列以供 LLM/自回归模型生成。现有方法要么用启发式（如 k-means on item embeddings）缺乏显式监督信号，要么用 GNN 但面临规模化问题（只利用局部图信息或无法处理工业级数据量）。用户行为上下文（协同过滤信号）与语义分词两个模块相互割裂，导致推荐质量欠佳。

**解决方案**: G2Rec（Graph-to-Rec）提出统一框架，将**全局图协同参与建模**（holistic graph co-engagement）与**有监督语义分词**融合：先用可扩展图方法从全局用户-物品共参与关系中提取用户兴趣原型（user interest prototypes），再用这些原型为 tokenization 提供显式监督信号，从而同时解决"局部图"与"无监督分词"两大痛点，并已在 Meta 多个产品线在线部署验证。

---

## 方法概述 / Method Overview

G2Rec 由三个核心模块构成：

### 1. 全局图协同参与建模（Holistic Graph Co-Engagement Modeling）
- 构建用户-物品交互图，利用**全局**共参与（co-engagement）关系而非局部 GNN 邻域来推断用户兴趣原型
- 设计可扩展图算法：避免传统 GNN 在工业规模下的 O(n²) 问题，支持数亿级节点
- 输出：每个物品的"兴趣原型分配"（interest prototype assignment），无需真实用户兴趣标注

### 2. 有监督语义分词（Supervised Semantic Tokenization）
- 基于图推断的兴趣原型为分词过程提供监督信号
- 不再依赖 k-means 或随机初始化等启发式，而是以图信号作为软标签（soft supervision）
- 产出语义上更有区分度的物品 token 序列

### 3. 工业级生成式推荐骨干（Industrial Generative Rec Backbone）
- 在上述 token 序列之上训练 LLM 风格自回归模型做序列推荐
- 在 Meta 内部使用自有大规模用户行为数据训练，并在多个产品面（product surfaces）做 online A/B 部署

---

## 创新性分析 / Innovation Analysis

| 创新点 | 说明 | 可行性 |
|--------|------|--------|
| 全局图 vs. 局部 GNN | 用 holistic co-engagement（类似矩阵分解的全局信号）替代 GNN 邻域聚合，解决规模化问题 | ✓ 已有 Meta 工业部署验证 |
| 分词有监督 | 用图推断的兴趣原型作为分词监督，去除启发式依赖 | ✓ 思路清晰，Loss 可自然定义 |
| 统一框架 | 两个过去分离模块（图建模 + 分词）在同一框架端到端优化 | ✓ 工程上可行，已有在线实验 |

与相关工作的区别：
- **TIGER / ActionPiece**: 用 RQ-VAE / BPE 分词，无图监督
- **CoFiRec**: 粗到细分词，但仍局部图
- **GraphRec / NGCF**: GNN 推荐，未与生成式框架结合

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | G2Rec | Best Baseline | Delta |
|--------|------|-------|---------------|-------|
| Meta 内部多产品面 | Online A/B (未披露具体数值) | ✓ 全面超越 | — | significant |
| Public benchmarks (多个) | NDCG@10 / HR@10 | 优于 existing SOTA | — | 竞争性 |

> 注：由于是工业论文，在线指标不公开具体数值。公开数据集结果在论文中有详细对比表。

---

## 评分 / Scoring

| 维度 | 分值 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 23 | 30 | 图监督分词的统一框架，工业规模可行 |
| Experimental SOTA delta | 12 | 15 | 在线 A/B + 公开数据集双重验证 |
| Experimental quality | 12 | 15 | 多数据集对比 + 在线部署 |
| Efficiency | 7 | 10 | 图算法可扩展，但图构建仍有额外开销 |
| Generalization | 5 | 5 | 跨多个 Meta 产品面部署验证 |
| Domain relevance | 22 | 25 | 工业级电商推荐，直接与达人内容/商品推荐对齐 |
| **Total** | **81** | **100** | |

---

## 代码复现 / Code Reproduction

完整复现代码见 `code/G2Rec/`。

- `model.py` — G2Rec 完整模型（图协同建模 + 有监督分词 + 自回归推荐）
- `data.py` — 玩具数据集生成（接口与真实数据对齐）
- `train.py` — 训练脚本（含 RQ-VAE 分词器 + 自回归推荐器）
- `requirements.txt` — 依赖
