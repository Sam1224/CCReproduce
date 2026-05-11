# Who Decides What Is Harmful? Multi-Agent Personalised Content Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework |
| **arXiv ID** | [2605.01416](https://arxiv.org/abs/2605.01416) |
| **提交日期** | 2026-05-02 |
| **作者** | Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak |
| **机构** | 未完全披露（波兰研究机构） |
| **领域桶** | STRONG |
| **综合评分** | **64 / 100** |

---

## 方法概述 (Chinese)

在线平台内容有害性判断高度主观——同一内容对不同用户的"伤害感知"差异显著。传统内容审核依赖中心化、自上而下的规则，无法适应这种主观性。

本文提出基于 LLM 的**多智能体个性化推断框架**，通过为每位用户构建独特的敏感度档案来过滤内容。框架包含三类智能体：

1. **Expert Agents（领域专家智能体）**：针对不同类型有害内容（暴力、仇恨言论等）的专域分析；
2. **Manager Agent（管理者智能体）**：协调内容分析流程和智能体选择；
3. **Ghost Profile Agent（幽灵档案智能体）**：模拟用户视角，将用户敏感度档案转化为决策输入。

与一系列非个性化基线相比，该系统在准确率上提升高达 32%，表现出更强的与用户个人敏感度的对齐能力。

## Method Overview (English)

This paper proposes an LLM-based multi-agent personalised inference framework for content moderation. Expert Agents handle domain-specific harmful content types, a Manager Agent orchestrates analysis and agent selection, and a Ghost Profile Agent simulates user perspectives using individual sensitivity profiles. Evaluated against non-personalised baselines with up to 32% accuracy improvement.

---

## Story Arc

**传统集中式内容审核规则无法适应用户对"有害内容"感知的主观差异 → 多智能体个性化推断框架为每位用户构建敏感度档案，通过专家+管理者+幽灵档案三类智能体协作，将内容审核决策与个人敏感度对齐，准确率提升32%。**

> "Harmful" is in the eye of the beholder. What's graphic violence for a child is acceptable news for a journalist. This framework builds per-user sensitivity profiles and uses a multi-agent LLM system to apply them at moderation time.

---

## 创新性分析

1. **个性化内容审核范式**：从平台级统一规则转向用户级个性化是有价值的研究方向；
2. **Ghost Profile Agent**：将用户敏感度档案转化为 LLM 可推理的视角输入，设计巧妙；
3. **多智能体协作架构**：专家分工+管理者协调的架构在可扩展性上有优势。

**局限性**：个性化信息收集的隐私问题；与电商合规审核（更偏向平台统一标准）的适配性待验证。

**与电商内容治理的关联**：达人（KOL/influencer）内容审核中，需要平衡平台规范与内容多样性，个性化审核提供了新视角。

---

## 关键指标 / Key Metrics

| 比较基线 | 指标 | 多智能体个性化框架 | 非个性化基线 |
|---|---|---|---|
| 多种非个性化基线 | 准确率 | **+32% 最高提升** | 基线 |
| — | 用户敏感度对齐 | 显著更优 | 较低 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 18 | 个性化审核框架有新颖性，但多智能体LLM架构非首创 |
| 实验SOTA增益 (SOTA delta) | 15 | 9 | 32%准确率提升，与非个性化基线比较 |
| 实验质量与消融 (Quality) | 15 | 8 | 与非个性化基线对比，需更严格对照 |
| 效率 (Efficiency) | 10 | 5 | 多智能体调用成本较高 |
| 泛化性 (Generalization) | 5 | 3 | 有限场景验证 |
| 领域相关性 (Domain) | 25 | 21 | 内容治理、审核政策，与电商内容生态高度相关 |
| **总分** | **100** | **64** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.01416
