# Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework |
| **作者** | Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak |
| **机构** | Not yet fully confirmed |
| **arXiv ID** | [2605.01416](https://arxiv.org/abs/2605.01416) |
| **提交日期** | May 2, 2026 |
| **代码** | Not yet public |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** Traditional content moderation systems rely on centralized, top-down rules that apply uniformly to all users — a single policy decides what everyone can see. This fails to accommodate the subjective, context-dependent nature of harm perception: what one user finds deeply offensive, another considers humorous or informative.

The paper proposes a **multi-agent personalized inference framework** that filters content based on individual user sensitivity profiles. The framework consists of three agent types:
- **Expert Agents:** Domain-specialized LLMs that evaluate content across specific harm dimensions (hate speech, misinformation, graphic violence, etc.).
- **Manager Agent:** Orchestrates agent selection and aggregates multi-expert signals, deciding which experts to activate for a given content type.
- **Ghost Profile Agent:** Simulates the target user's perspective using their historical preferences and sensitivity markers, adjusting the final moderation decision to match their individual tolerance profile.

The system achieves **personalized** moderation without requiring per-user model fine-tuning — preferences are encoded as profiles processed at inference time.

**ZH:** 框架用三类智能体实现个性化内容审核：**专家智能体**（领域专属 LLM，评估特定危害维度）、**管理智能体**（编排专家激活和信号聚合）、**幽灵画像智能体**（基于用户历史偏好模拟其敏感度，调整最终决策）。无需为每个用户微调模型，通过推理时画像实现个性化。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 内容审核平台采用统一规则，忽视了用户对"有害内容"的主观差异性感知（例如，10 岁用户和成年研究者对同一内容的容忍度截然不同）。
>
> **我们的解决方案:** 多智能体框架将用户敏感度画像融入审核决策，使同一平台可以为不同用户提供差异化的内容过滤，同时保持审核流程的可解释性和可审计性。

---

## 创新性分析 / Innovation Analysis

1. **个性化审核范式：** 将用户敏感度纳入审核决策，而非全局统一，具有真实商业价值（尤其是家长控制、青少年保护场景）。
2. **无需微调：** 画像在推理时注入，避免了为每个用户重训模型的高昂成本。
3. **多专家架构：** 专家分工专注不同危害维度，比单一通用模型更有针对性。
4. **vs. 先前工作：** 现有个性化推荐系统有类似思路，但审核领域尚无系统化的个性化框架。

---

## 关键指标 / Key Metrics

| Setting | Metric | Proposed | Best Non-Personalised |
|---------|--------|----------|----------------------|
| Personalised moderation | Accuracy | best+32% | — |
| User alignment | Correlation | ~0.71 | ~0.54 |

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 20/30 | 个性化审核框架思路好，但多智能体组合方式已有先例 |
| Experimental SOTA delta | 10/15 | +32% 准确率提升显著，但基线选取较弱 |
| Experimental quality | 10/15 | 评估数据集规模未知，需更多消融 |
| Efficiency | 5/10 | 多智能体调用增加延迟 |
| Generalization | 4/5 | 跨多种危害类别验证 |
| Domain relevance | 19/25 | 适用于电商平台个性化内容保护（未成年保护、差异化展示） |
| **Total** | **68/100** | |
