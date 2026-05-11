# Algospeak, Hiding in the Open: The Trade-off Between Legible Meaning and Detection Avoidance

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Algospeak, Hiding in the Open: The Trade-off Between Legible Meaning and Detection Avoidance |
| 作者 | Jan Fillies, Ronald E. Robertson, Jeffrey Hancock 等 |
| 机构 | 未完全披露 |
| arXiv | https://arxiv.org/abs/2605.06619 |
| 提交日期 | 2026-05-07 |
| 领域标签 | 内容审核 · 语言规避 · Algospeak · LLM检测 · 平台治理 |
| 桶类别 | STRONG |
| 综合评分 | **68 / 100** |

---

## 方法概述 (中文)

随着 LLM 在内容生成和内容审核中的广泛应用，创作者越来越多地使用 **Algospeak**（算法语）——一种有意修改措辞以规避自动内容审核算法的语言策略（如用"unalive"替代"suicide"，用符号或谐音绕过关键词过滤）。这在 TikTok、Instagram 等平台的达人创作者生态中尤为普遍。

本文对 Algospeak 背后的博弈动态进行了系统形式化：
1. **MUM (Majority Understandable Modulation) 概念**：定义了最优逃逸点——在该调制水平上，进一步的规避变形会提升检测逃脱率，但代价是受众无法理解原意，从而失去传播价值。
2. **可复现框架**：构建 700 个标注样本，覆盖 20 个基础句子 × 5 个调制级别 × 7 种规避策略。
3. **双重评估**：用 7 个主流 LLM 分别进行意义恢复测试（测理解度）和分类检测测试（测可检测性），系统量化逃逸-理解 trade-off。

结果揭示：轻度调制即可显著降低检测率，而语义保留在中等调制水平后快速劣化——存在明确的"甜蜜点"可供规避者利用。

---

## 故事线 (Story Arc)

> **现状不足：** 内容审核平台长期与规避者进行"猫鼠游戏"，但缺乏对 Algospeak 动态规律的形式化理解，导致防御策略被动应对、难以前瞻布防。
>
> **我们的解法：** 提出 MUM 概念框架并构建可复现基准数据集，定量刻画规避能力与可读性之间的 trade-off 曲线，为内容审核系统设计提供理论依据——Algospeak Modeling Framework。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | MUM 概念首次形式化了 Algospeak 最优规避点；可复现基准数据集 |
| vs. 先前工作 | 先前研究多为个案描述或单一平台观察；本文提供跨策略、跨LLM的系统性量化分析 |
| 实用价值 | 为平台构建鲁棒内容审核系统提供"攻击方知识"，有助于构建对抗数据集 |
| 局限 | 样本量（700 items）相对有限；英文为主，中文 Algospeak（如谐音字、拼音绕过）未涵盖 |

---

## 关键指标

| 实验 | 发现 |
|------|------|
| 轻度调制（Level 1-2） | 检测率大幅下降，语义保留 > 90% |
| 中等调制（Level 3） | 接近 MUM 甜蜜点，逃逸率高，理解度降至约 60% |
| 高度调制（Level 4-5） | 检测几乎完全逃逸，但受众理解度 < 30% |
| 7 个 LLM 的检测一致性 | 较弱，不同模型对相同规避策略的检测率差异显著 |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 18 | 30 | MUM概念新颖，但偏分析性研究，工程创新有限 |
| Experimental SOTA delta | 7 | 15 | 非SOTA比较型，分析框架，贡献是量化insight |
| Experimental quality | 10 | 15 | 7 LLMs × 5级别 × 7策略，框架严谨 |
| Efficiency | 4 | 10 | 轻量分析框架 |
| Generalization | 3 | 5 | 5种规避策略，可扩展 |
| Domain relevance | 22 | 25 | 直接关联达人/创作者规避内容审核，平台治理高相关 |
| **Total** | **68** | **100** | |
