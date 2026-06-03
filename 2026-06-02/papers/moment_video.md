# Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events |
| **Authors** | Xiaolin Liu, Yilun Zhu, Xiangyu Zhao, Xuehui Wang, Yan Li, Xin Li, Haoyu Cao, Xing Sun, Shaofeng Zhang, Xu Yang, Zhihang Zhong, Xue Yang |
| **Affiliations** | (Multiple institutions; includes ByteDance Seed based on Seed-2.0-Pro being best) |
| **Link** | https://arxiv.org/abs/2606.02522 |
| **Submission Date** | 2026-06-01 (appears in 2026-06-02 listing) |
| **Domain Bucket** | **STRONG** — video MLLM temporal understanding; highly relevant for short video content analysis |
| **Total** | **64 / 100** |

---

## 方法概述 / Method Overview

### 问题背景 (Problem)
视频 MLLM 在全局场景理解上进步迅速，但它们是否能**保留短暂关键视觉证据**（Momentary Visual Events）从未被系统测试。

许多实际问题的答案取决于"瞬时视觉事件"——仅持续几帧的局部动作或状态变化。这类证据可能：
- 被稀疏帧采样**跳过**
- 被视觉 token 压缩**抑制**
- 被粗粒度时间聚合**稀释**

> X is insufficient: Existing video benchmarks test global scene understanding — none specifically probe whether models can notice, count, describe, or reason about **momentary** visual evidence.

### 设计 (Design — Moment-Video Benchmark)
**规模**：1,000 条人工核验的视频 QA 对，覆盖：
- **7 大场景**（如运动、烹饪、社交互动等）
- **25 细粒度子类别**
- **4 种任务类型**：
  1. **Temporal Occurrence**：某事件是否发生？何时发生？
  2. **Temporal Counting**：某动作发生了多少次？
  3. **Action Description**：描述某时刻的动作
  4. **Temporal Reasoning**：基于时间顺序推理

**设计原则**：
- 每个 QA 对都锚定于一个**局部、可视察、采样敏感**的事件
- 模型不能依赖持久性物体、全局场景上下文或语言先验获得高分

---

## 故事弧 / Story Arc

> *"Video MLLMs score well on global scene understanding — but many real-world decisions hinge on events lasting only a few frames. Sparse sampling, visual token compression, and coarse temporal aggregation all conspire to erase this evidence. Moment-Video is the first benchmark specifically designed to expose this blind spot, requiring models to notice, count, or reason about transient evidence that cannot be recovered by language-side priors."*

---

## 创新点 / Innovation

1. **"瞬时性"（Momentariness）作为第一公民**：现有基准（VideoMME, MVBench 等）测试全局/持续内容，Moment-Video 专注采样敏感的短暂事件
2. **4 类任务覆盖不同难度**：从"是否发生"到"计数"再到"因果推理"，层层递进
3. **25 子类别精细诊断**：可定位具体失败模式（如烹饪场景 vs 运动场景表现差异）
4. **33 模型的大规模评估**：包括 Seed-2.0-Pro、GPT-5.4、多个 Gemini 模型及开源模型

与相关工作比较：
| 基准 | 专注瞬时事件 | 采样敏感设计 | 多任务类型 | 计数任务 |
|------|------------|------------|----------|--------|
| VideoMME | ❌ | ❌ | 部分 | ❌ |
| MVBench | ❌ | ❌ | ✅ | ❌ |
| **Moment-Video** | ✅ | ✅ | ✅ | ✅ |

---

## 关键指标 / Key Metrics

| 模型 | Overall Acc |
|------|-------------|
| **Seed-2.0-Pro（最优）** | **39.6%** |
| GPT-5.4 | 26.9% |
| Gemini-series | 20.4%–26% |
| 大多数开源模型 | < 25% |

**任务难度排序**：Temporal Occurrence ≈ Action Description < Temporal Counting < Temporal Reasoning

---

## 打分明细 / Scoring Breakdown

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | 瞬时事件聚焦有明确的研究价值；基准设计精细 |
| Experimental SOTA Delta | 7 | 15 | 基准论文；最优模型仅39.6%说明巨大gap，但非模型论文 |
| Experimental Quality/Ablations | 13 | 15 | 33模型、1000 QA对、7域25子类、分任务分析 |
| Efficiency | 3 | 10 | 不适用 |
| Generalization | 4 | 5 | 7大场景，广泛 |
| Domain Relevance | 17 | 25 | 短视频商品展示（操作细节）、直播审核（关键动作瞬间）的视频理解需求 |
| **Total** | **64** | **100** | |

---

## 与电商/内容治理的关联

- **商品短视频质量评估**：判断主播是否完成了某个动作（如展示商品背面），答案往往藏在几帧之中
- **直播违规检测**：许多违规行为（如短暂展示违禁商品、一闪而过的虚假宣传文字）需要模型捕捉瞬时帧证据
- **达人行为分析**：统计主播在视频中展示商品的次数、持续时长等，都是计数类/时间推理类任务
- 该基准揭示了当前最强 MLLM（Seed-2.0-Pro 仅 39.6%）在这些关键任务上的显著不足，为专用内容审核模型的设计提供了量化目标
