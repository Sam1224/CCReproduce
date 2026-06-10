# SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research |
| **Authors** | Pu Ning, Quan Chen, Kun Tao, Xinyu Tang, Tianshu Wang, Qianggang Cao, Xinyu Kong, Zujie Wen, Zhiqiang Zhang, Jun Zhou |
| **Affiliations** | Tsinghua University, Peking University, Ant Group, Renmin University of China |
| **arXiv** | [2606.09730](https://arxiv.org/abs/2606.09730) |
| **Submitted** | 2026-06-08 |
| **Venue** | ICML 2026 SCALE Workshop (Oral) |
| **Keywords** | Agentic LLM、委派智能、深度研究、上下文窗口、多智能体 |
| **Bucket** | WEAK |

---

## 方法概述 / Method Summary

在长程深度研究任务中，LLM 的上下文窗口是硬性瓶颈。SearchSwarm 提出**委派智能（Delegation Intelligence）**范式：主 Agent 将复杂任务分解为子任务，委派给专用子 Agent 执行，子 Agent 只返回摘要结果而非完整上下文，从而保护主 Agent 的上下文预算。

核心贡献：
1. **委派能力训练数据**：构建了一个覆盖"何时委派、委派什么、如何整合返回结果"的训练数据集，而自然语言语料中此类能力极为稀缺；
2. **SearchSwarm-30B-A3B**：基于 30B MoE 模型（激活参数 3B）微调的委派智能模型，在 BrowseComp 上达到 68.1，在 BrowseComp-ZH 上达到 73.3，是同等规模中最优；
3. **开源生态贡献**：重点填补开源社区在委派智能上的空白。

---

## 故事弧 / Story Arc

> **现状不足** → **提出方案**

现有 LLM Agent 在面对超出上下文窗口的长程任务时，要么截断信息，要么依赖人工管理上下文，缺乏自主委派子任务的能力。商业模型对委派智能有一定支持，但开源社区几乎没有专门的训练方案。

SearchSwarm 通过构建委派训练数据并微调开源模型，弥补这一能力缺口，使开源模型在复杂深度研究任务上与商业模型竞争。

---

## 创新性分析 / Innovation

| 维度 | 分析 |
|------|------|
| 委派智能操作化 | 将"何时/如何委派"定义为可训练能力，而非临时工程规则 |
| 数据构建 | 指出委派场景在自然语料中极稀缺，自建训练数据 |
| 结果 | 30B-A3B 规模在 BrowseComp 上超过所有同规模开源模型 |

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | SearchSwarm-30B-A3B | 对比最优同规模开源模型 |
|--------|------|---------------------|----------------------|
| BrowseComp | Score | 68.1 | SOTA（同规模） |
| BrowseComp-ZH | Score | 73.3 | SOTA（同规模） |

---

## 评分明细 / Score Breakdown

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 20 | 30 | 委派智能的操作化和训练方案新颖，但核心 Agent 架构依赖现有框架 |
| 实验指标 SOTA | 12 | 15 | BrowseComp 同规模 SOTA，有说服力 |
| 实验质量/消融 | 10 | 15 | 消融分析存在但有限 |
| 方法效率 | 7 | 10 | 委派设计保护主 Agent 上下文预算，效率改善明显 |
| 方法泛化性 | 3 | 5 | 主要在深度研究场景验证 |
| 论文相关性 | 16 | 25 | 可迁移用于内容治理 Agent 场景（如多 Agent 内容审核），间接相关 |
| **Total** | **68** | **100** | 强 Agent 论文，对内容生态 Agent 架构设计有参考价值 |
