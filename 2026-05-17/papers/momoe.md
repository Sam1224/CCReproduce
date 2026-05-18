# MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance |
| **Authors** | (University of Illinois Urbana-Champaign, scuba-illinois group) |
| **ArXiv** | https://arxiv.org/abs/2505.14483 |
| **Submitted** | May 2025; accepted EMNLP 2025 Main |
| **Venue** | EMNLP 2025 (ACL Anthology: 2025.emnlp-main.638) |
| **Code** | https://github.com/scuba-illinois/MoMoE ; see also `code/MoMoE/` |
| **Domain** | Content moderation, online governance, NLP, LLM |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: 内容审核系统通常为单一模型，难以跨社区、跨平台进行有效泛化，且无法提供可解释的审核决策。现有方法要么过度依赖大模型的高成本推理，要么缺乏对不同社区规范的精细适应性。  
> **我们的设计**: MoMoE（Mixture of Moderation Experts）是一个模块化、可扩展的混合专家框架，通过社区专用小模型专家的集成实现跨社区内容治理，并由 GPT-4o 提供可解释的后验决策说明。

### 四大算子 / Four Operators

MoMoE 将内容审核流程分解为四个可组合算子：

| 算子 | 功能 | 实现 |
|------|------|------|
| **Allocate** | 根据内容分配相关专家及权重 | RoBERTa-base 微调（社区分类7类 / 违规类别5类） |
| **Predict** | 各专家独立预测违规概率 | 轻量 LLM（Llama/Mistral 微调）专家集合 |
| **Aggregate** | 融合多专家预测 | 加权点积 or 多数投票 |
| **Explain** | 生成三级JSON格式的后验解释 | GPT-4o |

**两种实例化**:
- **MoMoE-Community**: 7个社区专用专家（按来源 subreddit 分类）
- **MoMoE-NormVio**: 5个规范违规类别专家（按违规类型分类）

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **架构设计** | 将 MoE（混合专家）范式引入内容审核，专家按社区/违规类型特化，解决了"一刀切"模型的泛化问题 |
| **可解释性** | Explain 算子生成结构化 JSON 说明，首次在审核框架层面集成解释机制 |
| **跨社区泛化** | 在 30 个未见过的 subreddit 上测试，验证了跨社区泛化能力 |
| **效率** | 主要推理由小型专家模型承担，GPT-4o 仅用于解释，成本可控 |
| **vs 先前工作** | 相比 Perspective API、单一 LLM 审核器，MoMoE 在解释性和跨社区精度上均有显著提升 |

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | MoMoE-Community | MoMoE-NormVio | 最强基线 |
|--------|------|-----------------|---------------|---------|
| Reddit (30 unseen subreddits) | Micro-F1 | **0.72** | 0.67 | ~0.70 (fine-tuned single model) |
| Recall | ↑ (↓ vs NormVio ~0.06) | Higher recall | — | — |
| Precision | Higher precision | — | — | — |
| Explanation quality | 3-level JSON | Reliable & concise | — | N/A |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 24 | 将 MoE 引入内容审核是新颖想法；可解释性设计有实用价值 |
| Experimental SOTA delta | 15 | 11 | 在30个未见社区上匹配或超越强基线，验证充分 |
| Experimental quality/ablations | 15 | 12 | 两种实例化对比，消融实验清晰 |
| Efficiency | 10 | 7 | 小专家+大解释器的架构合理，但多专家推理延迟仍存在 |
| Generalization | 5 | 4 | 跨社区泛化实验设计良好 |
| **Domain relevance** | **25** | **22** | 内容治理直接相关；与电商达人内容审核、平台违规检测高度匹配 |
| **Total** | **100** | **80** | EMNLP 2025 发表，方法实用，代码开源 |

---

## 代码复现 / Code Reproduction

复现代码位于 `code/MoMoE/`，实现了：
- 四算子框架（Allocate → Predict → Aggregate → Explain）
- RoBERTa-base Allocator 的训练与推理
- 轻量 SLM 专家模型（基于 Llama-3.2-1B 微调）
- 加权聚合与多数投票融合策略
- GPT-4o 解释生成接口（可替换为本地 LLM）
- 完整评估脚本（Micro-F1、Precision、Recall）
