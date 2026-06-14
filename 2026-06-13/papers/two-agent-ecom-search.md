# Iterating Toward Better Search: A Two-Agent Simulation Framework for Evaluating Agentic Search Architectures in E-Commerce

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Iterating Toward Better Search: A Two-Agent Simulation Framework for Evaluating Agentic Search Architectures in E-Commerce |
| **ArXiv ID** | [2606.12924](https://arxiv.org/abs/2606.12924) |
| **Authors** | Jetlir Duraj, Jayanth Yetukuri, Shuang Zhou, Dhruv Varma, Rui Kong, Ishita Khan, Qunzhi Zhou |
| **Affiliation** | eBay Inc. |
| **Submitted** | 2026-06-11 |
| **Source** | HuggingFace June 13 daily listing |
| **Bucket** | STRONG — 电商对话购物、Agent 评测、仿真框架 |

---

## 方法概述 / Method Overview

**故事弧线：** 对话购物助手上线前难以低成本覆盖足够多的用户类型；单一 Agent 仿真会产生不真实的"自我对话"。→ eBay 提出双 Agent 可替换仿真框架：buyer agent 固定（persona/mission/patience），responder agent 接真实搜索 API 可插拔替换，实现同一对话场景下的架构对比与失败归因。

**框架设计：**
- **Buyer Agent**：按 persona（14 个类型 bucket）、mission 和 patience 参数自主提出需求并执行行动，模拟真实用户分布；
- **Responder Agent**：调用真实 eBay 搜索 API 生成搜索结果页（SRP）和对话建议，可替换为不同架构（记忆策略/LLM backbone/judge）；
- **Failure Taxonomy**：系统化失败分类，将低分对话转化为具体修复点，驱动快速迭代（near-failure 减少 62%）；
- **Judge 分析**：实证发现 Gemini 与 Claude judge 在关键指标上存在显著分歧，对工业评测治理具有直接启示。

**关键发现：** 简单的 rolling-window memory 在多项指标上优于复杂的 intent-extraction memory，同时延迟降低 35%。

---

## 关键指标 / Key Metrics

| 系统 | Mission Success | SRP Relevance | CHAT Helpfulness | Latency |
|------|----------------|---------------|-----------------|---------|
| Sys-A（intent tracking）| base | base | base | base |
| **Sys-B（rolling-window）**| **+0.10** | **+0.07** | **+0.08** | **-35%** |

数据集：2,011 段对话，14 个 persona bucket；Gemini judge。  
失败率（Sys-B+）：near-failure 65→25（-62%），catastrophic 14→9（-36%）。

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 21 | 30 | 可插拔双 Agent 仿真 + 系统化失败归因，工程创新实用 |
| 实验指标 | 10 | 15 | 多 persona 覆盖，模拟数据含真实 API |
| 实验质量 | 12 | 15 | Judge 分歧分析、latency、failure taxonomy 三维度验证 |
| 方法效率 | 8 | 10 | rolling-window 方案更快更好，实用性高 |
| 方法泛化性 | 3 | 5 | 框架通用但数据/API 强依赖 eBay |
| 领域相关性 | 24 | 25 | 电商对话购物助手评测，含 memory/agent/评测治理 |
| **Total** | **78** | **100** | |

---

## Story Arc

> **现状不足：** 对话购物助手架构对比需要真实用户覆盖，成本高且无法控制变量；自我对话仿真失真。  
> **解法：** Buyer固定/Responder可插拔 → 同一场景复放对比架构 → failure taxonomy 归因 → 迭代验证"简单 memory 胜复杂方案"。
