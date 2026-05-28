# Improving Labeling Consistency with Detailed Constitutional Definitions and AI-Driven Evaluation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Improving Labeling Consistency with Detailed Constitutional Definitions and AI-Driven Evaluation |
| **Authors** | Konstantin Berlin, Adam Swanda |
| **Affiliations** | Cisco |
| **arXiv** | [2605.24247](https://arxiv.org/abs/2605.24247) |
| **Submitted** | 2026-05-22 |
| **Bucket** | STRONG |
| **Total** | **75 / 100** |

---

## 方法概述 / Method Overview

### EN
Automated content moderation pipelines rely on human-annotated golden labels, but short category specifications are insufficient for consistent labeling — annotators resort to intuition when edge cases are not explicitly addressed, causing label drift. This paper proposes an **AI-driven constitutional labeling workflow**: for each labeling category, a frontier LLM is first used to write a *detailed constitutional definition* that covers all edge cases in the category specification. The LLM then acts as the annotator, interpreting the constitution to produce the golden label. The key insight is that LLMs can hold much longer and more detailed specifications in their "working memory" than human annotators, and their responses are more consistent across instances. The authors evaluate on **HarmBench** across three content moderation categories (harassment, hate speech, non-violent crime) and compare against human annotators reading both short-form and long-form definitions.

### ZH
自动内容审核流水线依赖人工金标签，但简短的类别定义不足以保证标注一致性——标注者在遇到边界案例时依赖直觉，导致标签漂移。本文提出 **AI 驱动的宪法式标注工作流**：对每个标注类别，先用前沿 LLM 生成覆盖所有边界情形的**详细宪法定义**，再由 LLM 作为标注者解释宪法并生成金标签。核心洞见：LLM 可在"工作记忆"中保持比人类标注者更长更详细的规范，且跨实例的一致性更高。在 HarmBench 的骚扰、仇恨言论、非暴力犯罪三个内容审核类别上评估，与阅读短定义/长定义的人类标注者对比。

---

## 故事弧 / Story Arc

> **"短规范导致人工标注不一致"** → 用 LLM 撰写详细宪法定义，再由 LLM 充当标注者，比人类读同样文档更一致，与专家裁决对齐更好，跨模型不一致率降低 57×。

---

## 创新性分析 / Innovation Analysis

1. **宪法式规范 + LLM 标注的组合**：将 Constitutional AI 的思路应用于数据标注管道，而非模型对齐，应用场景新颖。
2. **跨模型不一致性作为规范质量诊断**：如果多个 LLM 对同一输入不一致，说明规范存在歧义——这是一个可量化的规范质量指标。
3. **扩展性**：一旦宪法写好，可以无限扩展标注规模，无需人工。
4. **与专家对齐**：LLM 标注与人类专家裁决的一致性优于普通标注者，说明 LLM 能捕捉细粒度规范意图。

---

## 关键指标 / Key Metrics

| Comparison | Metric | Constitutional LLM | Human Annotator |
|-----------|--------|-------------------|----------------|
| Cross-model inconsistency | rate | **57× lower** than paragraph definition | baseline |
| Expert adjudication alignment | agreement | **higher** than any human group | lower |
| Three LLMs unanimous agreement | rate | **higher** than 3 human annotators | lower |

*Evaluated on HarmBench: harassment, hate speech, non-violent crime categories*

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 22 | Constitutional AI 迁移至数据标注，诊断工具（跨模型不一致性）新颖 |
| Experimental SOTA delta | 15 | 10 | 57× 不一致率降低数字显著；与专家对齐更好 |
| Experimental quality / ablations | 15 | 11 | HarmBench 标准数据集，多模型对比 |
| Efficiency | 10 | 6 | 需要 LLM 推理成本，但可批量化 |
| Generalization | 5 | 4 | 适用于任何分类标注任务 |
| Domain relevance | 25 | 22 | 内容审核标注、达人违规审核、电商内容质量打标 |
| **Total** | **100** | **75** | |

---

## 电商治理价值 / E-commerce Governance Value

- **达人内容违规审核标注**：解决复杂违规类别（虚假宣传、违禁品推广）人工标注不一致问题
- **内容平台规范维护**：平台审核规范的宪法化写法可降低审核员理解歧义
- **大规模自动标注**：与人工相比成本大幅降低，可快速扩展标注规模
- **标注质量诊断**：跨模型不一致性作为规范质量自动诊断工具
