# Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents |
| **Authors** | Tianxiang Fei, et al. |
| **Affiliations** | Tencent |
| **arXiv** | [2606.09483](https://arxiv.org/abs/2606.09483) |
| **Submitted** | 2026-06-08 |
| **Keywords** | LLM Agent、记忆系统、双过程理论、自进化、Tencent |
| **Bucket** | WEAK |

---

## 方法概述 / Method Summary

现有 LLM Agent 记忆系统存在两个问题：一是记忆是静态的（Recall Only），缺乏动态更新和演化；二是缺乏认知科学层面的双过程结构（快速直觉 + 慢速推理）。

本文受认知科学**双过程理论（System 1/System 2）**启发，提出：
1. **System 1（快速记忆）**：基于 embedding 相似度的快速检索，适合频繁发生的模式匹配；
2. **System 2（慢速记忆）**：基于结构化推理的深度理解，适合罕见但重要的信息；
3. **自进化机制**：Agent 通过任务反馈自动更新两类记忆，实现长期知识积累。

---

## 故事弧 / Story Arc

> **现状不足** → **提出方案**

现有 RAG 式 Agent 记忆是"查找"的（retrieve-only），无法随任务经验演化；记忆检索是扁平化的，无法区分高频模式与深度推理所需信息。

双过程记忆系统对不同信息类型使用不同存储和检索机制，并通过反馈驱动记忆更新，使 Agent 能够随使用时间自我改进。

---

## 关键指标 / Key Metrics

| 数据集/任务 | 指标 | Dual-Process | RAG-only |
|-------------|------|--------------|---------|
| 多步骤推理任务 | 成功率 | +提升 | — |
| 长期任务（多轮） | 性能保持 | 更稳定 | — |

---

## 评分明细 / Score Breakdown

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 16 | 30 | 双过程理论应用于 Agent 记忆有新意，但框架偏概念层 |
| 实验指标 SOTA | 9 | 15 | 实验结果有限，任务较窄 |
| 实验质量/消融 | 8 | 15 | 消融分析基础 |
| 方法效率 | 7 | 10 | System 1 快速路径效率优化明确 |
| 方法泛化性 | 3 | 5 | Agent 记忆通用框架，理论泛化性好 |
| 论文相关性 | 12 | 25 | 可迁移到内容治理 Agent 的记忆设计，间接相关 |
| **Total** | **55** | **100** | 腾讯 Agent 基础研究，双过程记忆设计有参考价值 |
