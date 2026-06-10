# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | (Ant Group, Alibaba) |
| **Affiliations** | Ant Group / Alibaba |
| **arXiv** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | 2026-06-04 |
| **Keywords** | 电商查询推荐、Agent、chain-of-retrieval、GRPO、查询-商品对齐 |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

电商搜索中的**查询推荐（Query Recommendation）** 旨在主动为用户推荐潜在查询词，但现有方法存在"查询点击率高但商品转化率低"的脱节问题——生成的查询虽然看起来相关，但指向的商品库存与用户意图不符。

QueryAgent-R1 提出**记忆增强 Agentic 框架**，以**chain-of-retrieval 优化**为核心：
1. **Chain-of-Retrieval**：查询生成过程中，模型依次检索实际商品库存，通过"生成→检索→验证→修正"的链式过程确保生成查询的商品可达性；
2. **一致性奖励（Consistency Reward）**：将查询相关性（CTR）和下游商品参与度（CVR）联合纳入奖励信号，使用 GRPO 进行策略优化；
3. **记忆模块**：缓存历史检索结果，减少重复检索开销，提升推理效率。

---

## 故事弧 / Story Arc

> **现状不足** → **提出方案**

现有查询推荐方法把查询生成和商品检索分开优化：生成器只关注查询语义，不知道库存现状；检索器只关注相关性，不知道用户意图。导致推荐查询虽然语言流畅，但对应的商品不满足用户需求（查询 CTR↑ 但转化 CVR↓）。

QueryAgent-R1 将查询生成与商品检索**端到端联合优化**：在训练时以真实库存检索作为中间步骤，通过一致性奖励强制要求生成的查询必须"可达高质量商品"，从根本上解决 CTR-CVR 脱节问题。

---

## 创新性分析 / Innovation

| 维度 | 分析 |
|------|------|
| vs. 传统查询推荐 | 将真实库存检索纳入生成循环，而非训练后独立评估 |
| vs. 标准 GRPO | 一致性奖励将查询相关性+商品参与度联合考量 |
| vs. 纯 LLM 查询生成 | 记忆模块提升推理效率，适合工业级部署 |
| 对齐 | chain-of-retrieval 类比 chain-of-thought，将"检索感知"带入生成 |

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | QueryAgent-R1 | Baseline |
|-------------|------|---------------|----------|
| 电商查询推荐（离线） | Query CTR | +提升 | — |
| 电商查询推荐（离线） | Product CVR | +显著提升 | — |
| 在线 A/B 测试 | 综合查询参与度 | +有效提升 | — |

---

## 评分明细 / Score Breakdown

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 19 | 30 | chain-of-retrieval 思路新颖，但依赖现有 GRPO 框架 |
| 实验指标 SOTA | 11 | 15 | CTR+CVR 双指标提升有意义，但细节有限 |
| 实验质量/消融 | 10 | 15 | 有消融实验但覆盖面有限 |
| 方法效率 | 6 | 10 | 记忆模块提升效率，但 Agentic 框架整体较重 |
| 方法泛化性 | 3 | 5 | 专注电商查询推荐场景 |
| 论文相关性 | 29 | 25 → 25 | 电商查询推荐核心场景，直接命中 |
| **Total** | **74** | **100** | 电商查询推荐直接场景，CTR-CVR 联合优化有实用价值 |
