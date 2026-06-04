# Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events |
| **Authors** | Xiaolin Liu, Yilun Zhu, Xiangyu Zhao, Xuehui Wang, Yan Li, Xin Li, Haoyu Cao, Xing Sun, Shaofeng Zhang, Xu Yang, Zhihang Zhong, Xue Yang |
| **Affiliation** | Shanghai Jiao Tong University; Shandong University; Southeast University; Tencent Youtu Lab |
| **arXiv** | https://arxiv.org/abs/2606.02522 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | Video MLLM Benchmark · Temporal Reasoning · Short Video Understanding |
| **Code** | — |

---

## 方法概述 / Method Overview

### 问题 / Problem
现有视频 MLLM 在通用和长视频理解上取得了显著进步，但对**瞬时视觉事件（momentary visual events）**的感知能力仍被低估和低研究：许多实际问题由仅持续几帧的短暂动作或状态转换决定（如商品瑕疵、危险动作、违规行为片段）。稀疏帧采样可能跳过关键证据，视觉 token 压缩可能抑制它，粗粒度时序聚合可能稀释它。

Existing video MLLMs have progressed on general/long-video understanding, but their ability to perceive **momentary visual events** — determined by localized actions or state transitions lasting only a few frames — remains underexplored. Sparse sampling may skip evidence; token compression may suppress it; temporal aggregation may dilute it.

### 方法 / Method

**Moment-Video 基准：**

1. **1,000 人工验证的 video-QA 对**，覆盖 7 个领域（sports, cooking, surveillance, etc.）和 25 个细粒度子类别。

2. **4 类任务类型：**
   - Temporal Occurrence（时序发生）
   - Temporal Counting（时序计数）
   - Action Description（动作描述）
   - Temporal Reasoning（时序推理）

3. **设计原则：** 每个问题均基于局部可观测、视觉可验证、采样敏感的事件，要求模型**注意、计数、描述或推理**瞬时证据，而非依赖持久物体、全局场景或语言先验。

**Story Arc:** "视频MLLM擅长理解长视频却"看不清"关键瞬间 → Moment-Video 专测瞬时时序忠实度"

---

## 关键指标 / Key Metrics

| Model | Overall Accuracy | Notes |
|-------|-----------------|-------|
| Seed-2.0-Pro | 39.6% | Best performing model |
| Most open-source models | <25% | Significantly below commercial |

**SOTA gap:** 39.6% best accuracy indicates massive room for improvement in temporal fidelity.

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 18 | Novel focus on momentary events; sampling-sensitive design; 4 task types |
| SOTA Delta | 15 | 10 | Clear baseline established; best model at 39.6% (large headroom) |
| Exp. Quality | 15 | 11 | 1000 verified QA pairs, 7 domains, 25 subcategories |
| Efficiency | 10 | 5 | Evaluation benchmark |
| Generalization | 5 | 4 | 7 diverse domains |
| Domain Relevance | 25 | 13 | Video content understanding relevant to e-commerce content ecosystem (product demos, live stream key moments) |
| **Total** | **100** | **61** | |

---

## 电商相关性 / E-commerce Relevance

瞬时视觉事件在电商内容中普遍存在：
- 商品展示中的关键细节（几帧内的商品特写）
- 直播违规行为的短暂片段（违规展示、虚假宣传动作）
- 商品视频中的效果展示（使用效果的关键帧）

Moment-Video 的评测体系可用于评估 MLLM 在电商短视频理解中对关键瞬间的感知能力。
