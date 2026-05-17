# E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs |
| **Authors** | (Research team — Taobao / Alibaba) |
| **Affiliation** | Alibaba (Taobao) |
| **arXiv ID** | [2602.08355](https://arxiv.org/abs/2602.08355) |
| **Submission Date** | February 9, 2026 |
| **Domain Tags** | `#e-commerce` `#benchmark` `#MLLM` `#short-video` `#commercial-intent` `#taobao` `#regulatory-compliance` |
| **Total** | **71 / 100** |

---

## 故事弧线 / Story Arc

**现有问题:** 电商短视频（带货视频）是一个以**转化为目标**、**多模态信号密集**的特殊领域。现有基准主要关注通用视频理解任务，忽略了商业意图推理——而商业意图是电商短视频区别于普通视频的核心。现有 MLLM 在此场景下的能力边界尚不明确。

**设计方案:** 构建 **E-VAds（E-commerce Video Ads Benchmark）** — 首个专为电商短视频理解设计的 MLLM 评测基准，提出**多模态信息密度评估框架**量化电商内容复杂度，并用多智能体系统生成高质量开放式 QA 对。

---

## 方法概述 / Method Overview

### 数据集构成

```
E-VAds Dataset
├── Source: Taobao (淘宝) platform videos
├── Scale: 3,961 high-quality e-commerce videos
│          × 5 task categories
│          = 19,785 open-ended QA pairs
│
└── Task Categories:
    ├── Basic Perception      — 视觉/音频基础感知
    ├── Cross-Modal Detection — 跨模态信号检测（文字+图像+语音一致性）
    ├── Marketing Logic       — 营销逻辑推理（卖点提取、USP分析）
    ├── Consumer Insight      — 消费者洞察（目标人群识别、需求挖掘）
    └── Regulatory Compliance — 合规检测（违规声明、虚假宣传）
```

### 多模态信息密度评估框架

核心发现：电商内容在视觉、音频、文字三个维度的信息密度均显著高于主流通用数据集，揭示了现有通用 MLLM 在电商场景的结构性短板。

### 多智能体 QA 生成
- 使用多智能体系统（专家角色分工）生成 QA 对
- 保证问题覆盖五大任务维度，平衡分布

---

## 关键指标 / Key Metrics

| Finding | Detail |
|---------|--------|
| Information density | E-commerce videos are substantially denser across visual, audio, and text modalities vs general datasets |
| Model performance | Significant gaps between e-com and general-domain performance for all evaluated MLLMs |
| Regulatory Compliance task | Most MLLMs perform significantly worse than human annotators |

---

## 创新性分析 / Innovation Analysis

- **首个专为电商带货短视频设计的 MLLM 基准**，填补领域空白
- 多模态信息密度评估框架揭示了电商视频的本质特征
- "Regulatory Compliance"维度直接对标电商内容合规检测需求
- 多智能体 QA 生成确保问题多样性与质量

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 19 | 30 | 首个电商短视频 MLLM 基准；信息密度评估框架新颖 |
| Experimental SOTA Delta | 9 | 15 | 基准论文；揭示多模型性能缺口 |
| Experimental Quality | 10 | 15 | 专家标注；多智能体生成；全面任务分类 |
| Efficiency | 5 | 10 | 基准论文，无效率贡献 |
| Generalization | 4 | 5 | 覆盖五类电商核心任务 |
| Domain Relevance | 24 | 25 | 电商带货短视频理解；含合规检测维度 |
| **Total** | **71** | **100** | |
