# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **ArXiv ID** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Authors** | Alibaba International Digital Commerce Group team |
| **Affiliation** | Alibaba International Digital Commerce Group |
| **Submitted** | 2026-06-04 |
| **Source** | WebSearch fallback (arXiv direct 403) |
| **Bucket** | STRONG — 电商查询推荐、Agent、RAG |

---

## 方法概述 / Method Overview

**故事弧线：** 现有查询推荐方法仅优化查询级别的相关性（CTR），忽略检索到的商品是否满足用户下游偏好（CVR），导致"查询点击率高但转化率低"的工业痛点。→ QueryAgent-R1 将查询生成与真实商品检索 grounding 结合，通过强化学习（Chain-of-Retrieval Optimization）同时优化相关性与下游参与度。

**核心模块：**
1. **Memory Abstraction Module**：高效的用户画像模块，将用户历史行为抽象为可检索记忆，支持个性化查询生成；
2. **Chain-of-Retrieval Optimization**：在 RL 训练的推理链中嵌入真实商品检索步骤，使 Agent 能根据实际检索结果验证并调整生成的查询；
3. **Consistency Reward**：专门设计的奖励函数，联合优化查询相关性（CTR 信号）与下游参与度（CVR 信号），解决指标解耦问题。

**关键创新：**
- 将"检索验证"内化为 RL 训练过程（training-time grounding），而非仅在推理时调用；
- 双目标一致性奖励弥合了 CTR/CVR 优化目标之间的 gap；
- Memory 模块实现轻量级实时个性化。

---

## 关键指标 / Key Metrics

| 指标 | QueryAgent-R1 | 传统方法 |
|------|---------------|---------|
| CTR (↑) | 显著提升 | baseline |
| CVR (↑) | 显著提升（核心贡献） | baseline（传统方法 CVR 常无改善） |
| 查询-商品对齐度 | 大幅改善 | 仅查询相关性优化 |

（具体数值为论文线上 AB 实验结果，未公开绝对值）

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 20 | 30 | 检索 grounding + 一致性奖励的组合新颖；Memory 模块设计实用 |
| 实验指标 | 10 | 15 | 线上 AB 有说服力，但具体数值未完全公开 |
| 实验质量 | 11 | 15 | 多组消融实验（memory/reward/retrieval grounding 各模块） |
| 方法效率 | 7 | 10 | Memory 抽象降低实时成本 |
| 方法泛化性 | 4 | 5 | 框架适用于其他平台的查询推荐 |
| 领域相关性 | 22 | 25 | 核心电商场景（Alibaba 国际），直接面向查询推荐/内容发现 |
| **Total** | **74** | **100** | |

---

## Story Arc

> **现状不足：** 传统查询推荐仅看 CTR，忽视检索到的商品是否真正转化，导致高点击低转化。  
> **解法：** 把真实商品检索嵌入 RL 训练链路（chain-of-retrieval）→ 一致性奖励联合优化 CTR+CVR → Memory 模块实现个性化 → 线上 AB 双指标共同提升。
