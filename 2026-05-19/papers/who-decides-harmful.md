# Who Decides What Is Harmful? Content Moderation via Multi-Agent Personalised Inference

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework |
| **Authors** | Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak |
| **arXiv** | https://arxiv.org/abs/2605.01416 |
| **Submitted** | May 2, 2026 |
| **Domain** | Content Moderation · Multi-Agent LLM · Personalization · Safety |

---

## 方法概述 / Method Summary

传统内容审核系统依赖中心化的"一刀切"规则，无法适应用户对有害内容感知的主观差异。本文提出基于 LLM 的**多智能体个性化推理框架**：

- **Expert Agents：** 特定领域专家 Agent，从专业角度分析内容
- **Manager Agent：** 协调多 Expert Agent 的分析，选择最相关的专家组合
- **Ghost Profile Agent：** 模拟目标用户的视角和价值观，使审核决策与个人敏感度对齐

框架根据每位用户的独特敏感性档案过滤内容，支持个性化的内容审核策略。

---

## 故事弧线 / Story Arc

> 中心化内容审核规则无法适应用户敏感性差异 → 三类专业 Agent 协同模拟用户视角做个性化判断 → 相比非个性化基线提升 32% 准确率，增强与用户价值观的对齐

---

## 创新分析 / Innovation Analysis

- Ghost Profile Agent 模拟用户视角是一个新颖的个性化审核思路
- 但核心任务（个性化内容安全）与电商/达人治理的场景对齐度有限
- 32% 准确率提升明显，但缺乏与 SOTA 系统的直接对比

---

## 关键指标 / Key Metrics

| Task | Metric | Result |
|------|--------|--------|
| Personalised content moderation | Accuracy vs non-personalised baselines | +32% |
| User sensitivity alignment | — | Improved |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 18 | 30 | Ghost Profile Agent 新颖，但整体框架较直观 |
| Experimental SOTA delta | 10 | 15 | +32% 准确率，与 SOTA 对比不够充分 |
| Experimental quality | 9 | 15 | 评测范围有限 |
| Efficiency | 5 | 10 | 多 Agent 调用成本未讨论 |
| Generalization | 3 | 5 | 通用内容审核，非电商专用 |
| Domain relevance | 16 | 25 | 内容治理相关但非直接电商/达人场景 |
| **Total** | **61** | **100** | |

