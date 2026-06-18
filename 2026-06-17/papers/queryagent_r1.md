# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| **Affiliations** | Alibaba International Digital Commercial Group |
| **arXiv** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | ~2026-06-05 |
| **Domain Tags** | e-commerce, query recommendation, agent, chain-of-retrieval, Alibaba International |
| **Code** | `code/QueryAgent-R1/` |

---

## 方法概述 / Method Summary

电商搜索中的查询推荐系统传统上只优化"查询级相关性"（CTR），忽视了被推荐查询检索到的商品是否真正符合用户下游转化偏好，导致高点击低转化的困境。QueryAgent-R1 提出**链式检索优化**（Chain-of-Retrieval Optimization）框架：将查询推荐定义为一个 Agent 任务，Agent 首先生成候选查询，然后主动调用商品检索器获取检索结果，再基于检索结果和用户偏好进行端到端打分与优化，实现查询生成-检索-评估的闭环对齐。结合记忆增强机制（Memory Augmentation）保留用户长期偏好。

**Story arc**: 现有方法孤立优化查询生成与商品检索，导致"被推荐查询"与"检索结果"之间的对齐鸿沟 → 设计 Agent 框架将检索结果纳入查询优化回路，实现端到端 CTR+CVR 联合提升。

**Key components**:
1. **Chain-of-Retrieval**: Agent 生成查询 → 实时检索 → 基于检索结果打分 → 优化查询生成
2. **Memory Augmentation**: 存储用户历史偏好，增强个性化
3. **R1-style RL Training**: 使用强化学习（类 R1/GRPO）训练 Agent，奖励信号来自下游转化指标
4. **End-to-End Alignment**: 直接以 CVR 信号优化 Agent 决策

---

## 创新性分析 / Innovation Analysis

**vs. prior work**:
- 传统查询推荐（SASRec 等序列推荐变体）不考虑检索器行为，QueryAgent-R1 将检索器纳入 Agent 决策回路
- R1-style RL 应用于查询推荐是较新的范式，将 LLM 推理能力迁移至电商搜索
- Alibaba International 线上验证，具有工业落地价值

**Novelty assessment**: 创新点清晰（检索-生成对齐），R1 强化学习 + Agent + 电商查询推荐的结合具有开创性，结果可信。

---

## 关键指标 / Key Metrics

| Dataset/System | Metric | QueryAgent-R1 | Baseline |
|---------------|--------|---------------|----------|
| Alibaba International (online A/B) | Query CTR | **+2.9%** | — |
| Alibaba International (online A/B) | Guided CVR | **+3.1%** | — |
| Offline datasets | Consistent improvements | yes | — |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 25 | 30 | 链式检索优化 + R1-style RL，查询-检索对齐新范式 |
| Experimental SOTA delta | 13 | 15 | 线上 A/B: +2.9% CTR, +3.1% CVR |
| Experimental quality / ablations | 14 | 15 | 线上线下双验证，阿里国际业务场景 |
| Efficiency | 7 | 10 | Agent 推理有开销，但离线预计算部分缓解 |
| Generalization | 3 | 5 | 仅阿里国际验证，其他平台待验证 |
| Domain relevance | 24 | 25 | 阿里国际电商查询推荐，直接电商场景 |
| **Total** | **86** | **100** | 工业 A/B 强验证，电商搜索推荐高度相关 |
