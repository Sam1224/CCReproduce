# Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework

## 基本信息 / Basic Info

| 字段 | 值 |
|------|-----|
| **Title** | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework |
| **arXiv** | https://arxiv.org/abs/2605.01416 |
| **Authors** | (Not retrieved; multi-institution) |
| **Affiliation** | (Not retrieved) |
| **Submitted** | May 2, 2026 |
| **Domain Tags** | `content-moderation` `multi-agent` `personalization` `policy` `governance` |
| **Code** | Not provided |
| **Bucket** | STRONG |
| **Total** | **60** |

---

## 方法概述 / Method Overview

**中文：**
该论文提出了一个基于 LLM 的**多智能体个性化推理框架**，用于内容审核决策。现有平台内容审核策略往往"一刀切"，忽视用户个体敏感度差异。该框架包含三类智能体：(1) **领域专家智能体（Expert Agents）**——负责特定有害内容类别的分析（如暴力、成人内容、虚假信息等）；(2) **管理智能体（Manager Agent）**——协调内容分析和专家选择；(3) **幽灵档案智能体（Ghost Profile Agent）**——模拟个体用户视角（含年龄、文化背景、历史等），为审核决策提供个性化情境。最终输出基于个体用户敏感度档案的差异化审核决策。

**English:**
This paper proposes an LLM-based multi-agent personalised inference framework for content moderation. It goes beyond one-size-fits-all platform policies by incorporating individual user sensitivity profiles. The architecture combines domain-specific Expert Agents (for violence, adult content, misinformation, etc.), a Manager Agent (orchestrating content analysis and agent selection), and a Ghost Profile Agent (simulating individual user perspectives including age, cultural context, history) to produce personalised, context-aware moderation decisions.

---

## 故事线 / Story Arc

平台内容审核标准统一（如 Community Standards）难以反映个体用户的实际感受和容忍度差异 →  
平台治理正当性（legitimacy）依赖审核决策与用户期望的对齐 →  
多智能体框架：专家分析内容维度 + 幽灵档案模拟用户视角 → 个性化审核决策 →  
治理权力从"平台统一标准"向"用户敏感度自适应"转移。

---

## 创新性分析 / Innovation

- **核心创新**：将内容审核的"治理正当性"问题纳入 LLM agent 框架，通过幽灵档案智能体模拟个体用户视角，是内容治理领域的新视角。
- **工程可行性**：多智能体协调存在推理延迟问题，在电商平台实时审核场景下需要进一步优化。
- **与先前工作差异**：先前工作（LlamaGuard 等）关注准确率，本文关注治理正当性和个性化，角度不同。
- **局限**：更多是概念框架，缺乏强量化评估数据；个性化审核在监管合规场景下存在法律风险。

---

## 关键指标 / Key Metrics

| Setting | Metric | Value | Notes |
|---------|--------|-------|-------|
| Content moderation accuracy | Not specified | — | Focuses on legitimacy |
| User alignment | Qualitative | Demonstrated | Via Ghost Profile simulation |

---

## 评分 / Scoring

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 19 / 30 | 将治理正当性纳入 agent 框架是新视角；幽灵档案智能体概念有趣 |
| 实验指标 | 8 / 15 | 量化数据不充分，主要是定性框架展示 |
| 实验质量 | 9 / 15 | 框架验证较弱，缺乏强量化消融 |
| 效率 | 5 / 10 | 多 agent 推理开销较大 |
| 泛化性 | 3 / 5 | 概念通用但实际部署受限 |
| 相关性 | 16 / 25 | 内容治理框架，适用于电商达人内容个性化管理；但不直接面向违规检测的精度提升 |
| **Total** | **60** | |
