# Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling |
| **Authors** | Seojeong Park, Jiho Choi, Junyong Kang, Seonho Lee, Jaeyo Shin, Hyunjung Shim |
| **Affiliation** | Not fully retrieved |
| **arXiv** | https://arxiv.org/abs/2606.02578 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | MLLM Evaluation · Automated Judging · Content Quality · Reward Modeling |
| **Code** | — |

---

## 方法概述 / Method Overview

### 问题 / Problem
多模态 LLM 作为评判者（MLLM-as-a-Judge）时存在**感知判断偏差（Perceptual Judgment Bias）**：当视觉证据与文本线索冲突时，MLLM 倾向于奖励"看似合理的叙述"而非"感知上正确的答案"——即 judge 将注意力锚定在**响应文本**上，而非自身的视觉感知。

When used as judges (MLLM-as-a-Judge), multimodal LLMs exhibit **Perceptual Judgment Bias**: when visual evidence conflicts with textual cues, MLLMs tend to reward plausible narratives over perceptually correct answers — anchoring on **response text** rather than their own visual perception.

### 方法 / Method

**两步骤解决方案：**

**Step 1 — 感知扰动判断数据集（Perceptually Perturbed Judgment Dataset, PPJD）：**
通过**最小化编辑**（minimal edits）构建反事实响应，隔离感知错误并提供可验证的监督信号。这类数据专门设计为"只改变感知准确性而不改变文本流畅性"的样本对，迫使 judge 必须依赖视觉感知才能正确判断。

**Step 2 — 统一训练框架：**
结合 **GRPO 奖励**（Group Relative Policy Optimization，结构化奖励）和**批排序目标（batch-ranking objective）**，训练 MLLM judge 在感知上更可靠。

**Story Arc:** "MLLM-as-a-Judge 依赖文本而非视觉 → 用感知扰动数据集+GRPO奖励+批排序迫使 judge 真正'看'图"

*MLLM judges rely on text rather than vision → use perturbed dataset + GRPO reward + batch ranking to make judges truly "see" the image.*

---

## 创新性分析 / Innovation

1. **新现象发现**：首次系统定义和量化"感知判断偏差"，揭示了现有 MLLM judge 的根本性缺陷。
2. **PPJD 数据集**：最小化编辑 + 反事实设计提供了可验证的监督信号，解决了视觉评估标注成本高的问题。
3. **GRPO + 批排序**：将强化学习目标与排序损失结合，比单一监督学习更有效地校正感知偏差。
4. **实际应用价值**：对内容质量自动评估（图片、视频 caption 质量、广告审核）有直接指导意义。

---

## 关键指标 / Key Metrics

| Dataset | Metric | Trained Judge | Vanilla MLLM Judge |
|---------|--------|--------------|---------------------|
| PPJD (perturbed) | Perceptual Accuracy | Improved | Low |
| Visual QA Benchmarks | Judge Agreement | Improved | Biased toward text |

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 22 | Novel perceptual bias phenomenon; PPJD dataset design; GRPO+batch-ranking framework |
| SOTA Delta | 15 | 10 | Improvement in perceptual judging accuracy |
| Exp. Quality | 15 | 10 | New benchmark + ablation of GRPO vs. ranking objectives |
| Efficiency | 10 | 6 | Training framework; inference same as standard MLLM |
| Generalization | 5 | 3 | Visual-language domain |
| Domain Relevance | 25 | 17 | MLLM-as-judge quality assessment directly applies to automated content quality scoring and caption evaluation in e-commerce |
| **Total** | **100** | **68** | |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

MLLM-as-a-Judge 的感知偏差在以下场景直接影响质量：
- **商品图文质量评估**：自动评判商品图片与描述的一致性时，若 MLLM 倾向于文本而忽视图像，将导致虚假商品信息漏判
- **达人内容质量打分**：评估图文/视频内容质量时，需 judge 真正基于视觉内容而非仅凭文字描述
- **广告审核**：广告合规审核中视觉与文本不一致的案例检测

MLLM perceptual bias directly impacts: product image-text consistency evaluation (false product info detection), influencer content quality scoring, and ad compliance review (detecting visual-text inconsistency).
