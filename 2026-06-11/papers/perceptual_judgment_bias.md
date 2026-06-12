# Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling |
| **Authors** | Seojeong Park, Jiho Choi, Junyong Kang, Seonho Lee, Jaeyo Shin, Hyunjung Shim |
| **Affiliation** | KAIST |
| **ArXiv** | [2606.02578](https://arxiv.org/abs/2606.02578) |
| **Submitted** | June 2, 2026 |
| **Conference** | ICML 2026 |
| **Code** | https://github.com/kaist-cvml/perception-judge |
| **Domain Tags** | `MLLM` `evaluation` `LLM-as-Judge` `data-quality` `reward-modeling` |
| **Total** | **60 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 多模态 LLM-as-Judge（评判模型）存在"感知判断偏差"（Perceptual Judgment Bias）：当视觉证据与文本线索冲突时，评判模型倾向于奖励貌似合理的文本叙述而非感知正确的视觉答案。

**解决方案：** 构建感知扰动判断数据集（Perceptually Perturbed Judgment Dataset），并结合 GRPO 奖励和批次排序目标训练，校正评判偏差。

---

## 方法概述 / Method

1. **Perceptually Perturbed Judgment Dataset：** 通过最小化视觉编辑构建反事实（counterfactual）响应，隔离感知错误，提供可验证的监督信号。

2. **统一训练框架：**
   - **GRPO-based reward：** 基于感知正确性的奖励信号
   - **Batch-ranking objective：** 批次内响应排序目标，提升判别一致性

3. **应用场景：** 用于自动化内容质量评估、数据标注质量控制。

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| ICML 2026 | Perceptual accuracy | SOTA vs prior judges |
| Counterfactual eval | Judgment consistency | Significant improvement |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 18 | 30 | Counterfactual dataset construction + GRPO reward for perceptual bias is novel |
| Experimental SOTA delta | 10 | 15 | ICML 2026 acceptance validates quality |
| Experimental quality / ablations | 12 | 15 | ICML standard; controlled perturbation experiments |
| Efficiency | 4 | 10 | Not efficiency focused |
| Generalization | 4 | 5 | Applicable to various multimodal evaluation scenarios |
| Domain relevance | 12 | 25 | Relevant for automated content evaluation and labeling quality control |
| **Total** | **60** | **100** | |

---

## 中文摘要

KAIST 提出的 ICML 2026 工作，发现 MLLM 评判模型（LLM-as-Judge）在视觉和文本线索冲突时存在感知判断偏差——偏好文本叙述而非视觉真相。本文构建感知扰动判断数据集（通过最小化视觉修改生成反事实样本），并结合 GRPO 奖励和批次排序损失训练修正模型。对于电商内容质量评估和大规模数据标注质量控制场景，此方法可提升自动化评判系统的可靠性。
