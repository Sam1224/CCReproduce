# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| 作者 | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| 机构 | Alibaba International Digital Commercial Group（阿里巴巴国际数字商业集团）|
| arXiv | https://arxiv.org/abs/2606.05671 |
| 提交日期 | 2026-06-04 |
| 领域标签 | 电商搜索 · 查询推荐 · LLM Agent · 强化学习 · 用户记忆 |
| 桶类别 | STRONG |
| 综合评分 | **76 / 100** |

---

## 方法概述 (中文)

电商查询推荐（Query Recommendation）旨在主动向用户推荐与其潜在兴趣匹配的搜索查询词，引导用户进行后续搜索与购买。现有方法存在核心矛盾：**高 CTR、低 CVR**——推荐的查询词获得点击，但用户点进去搜出的商品与实际购买意图不符，导致转化率低。

**QueryAgent-R1** 的解法是将查询推荐建模为**记忆增强的链式检索智能体任务**：

1. **链式检索优化（Chain-of-Retrieval Optimization）**：查询生成不再只依赖语言模型输出，而是让 Agent 先生成候选查询词，再实际调用商品检索引擎（真实库存检索），根据检索到的商品集合验证并精炼查询词——使生成的查询词真实对应有效的商品库存。

2. **一致性奖励（Consistency Reward）**：在强化学习（RL）训练中设计一致性奖励信号：联合优化查询词语义相关性（CTR 代理）和下游商品参与度（CVR 代理），使 Agent 学习生成既会被点击、又能带来转化的查询词。

3. **记忆抽象模块（Memory Abstraction）**：维护用户偏好摘要的持久记忆，让 Agent 每次推荐时能快速检索历史偏好画像，实现个性化查询推荐而不需对全量历史重复计算。

---

## 故事线 (Story Arc)

> **现状不足：** 电商查询推荐"高点击、低转化"——推荐词语义相关但库存不匹配，或推荐词只考虑用户短期兴趣不考虑下游商品质量。
>
> **我们的解法：** QueryAgent-R1 让查询生成 Agent 先生成再检索再修正（Chain-of-Retrieval），用一致性 RL 奖励联合优化 CTR 与 CVR，并用记忆模块沉淀用户偏好，从而实现查询词与商品库存、用户意图的三方对齐。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | Chain-of-Retrieval：查询生成→实时检索→基于检索结果精炼，将商品库存引入查询生成回路 |
| 奖励设计 | 一致性奖励联合优化 CTR+CVR，解决高点击低转化问题 |
| vs. 先前工作 | 传统方法仅优化查询文本相关性；R1 系列（类 DeepSeek-R1）RL 思路迁移至查询推荐 |
| 局限 | Chain-of-Retrieval 增加推理延迟；记忆模块持久化设计未详述 |

---

## 关键指标

| 实验 | 数据集/场景 | 指标 | QueryAgent-R1 | Baseline |
|------|------------|------|---------------|---------|
| 离线评估 | 阿里国际电商数据集 | CTR+CVR 综合 | 优于传统查询推荐方法 | — |
| 线上 A/B | 阿里国际电商 | 综合收益 | 正向（数值未公开）| — |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | Chain-of-Retrieval 思路新颖；记忆模块有一定工程创新 |
| Experimental SOTA delta | 11 | 15 | 有线上 A/B 验证，但具体数值未公开 |
| Experimental quality | 11 | 15 | 离线+在线双验证；消融实验细节待核实 |
| Efficiency | 7 | 10 | 记忆抽象减少重复计算；链式检索增加延迟 |
| Generalization | 4 | 5 | 查询推荐通用框架，理论上可迁移 |
| Domain relevance | 23 | 25 | 直接针对电商搜索查询推荐，阿里国际平台实战 |
| **Total** | **76** | **100** | |

---

## 相关链接

- arXiv: https://arxiv.org/abs/2606.05671
- 机构: Alibaba International Digital Commercial Group
