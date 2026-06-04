# PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning |
| **Authors** | (from arXiv:2606.02443; full author list not retrieved due to 403) |
| **Affiliation** | Not fully retrieved; paper submitted June 1, 2026 |
| **arXiv** | https://arxiv.org/abs/2606.02443 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | Video Safety · Content Governance · MLLM Benchmark · Violation Detection |
| **Code** | `code/PaSBench/` |

---

## 方法概述 / Method Overview

### 问题 / Problem
现有视频安全评估体系聚焦于静态内容分类，无法模拟真实场景中的**流式视频安全预警**需求。在危险事件即将发生前，存在一个可供干预的时间窗口——如何在该窗口内由视频MLLM发出及时、精准的安全警告，是当前工业级内容监控系统亟需解决的核心挑战。

Existing video safety evaluation focuses on static classification and does not simulate real-world **streaming proactive safety warning** requirements. Between the first visible sign of danger and when an accident occurs, there is a window for intervention. How to issue timely, calibrated safety warnings within this window by video MLLMs is a core challenge for industrial content monitoring systems.

### 方法 / Method
PaSBench-Video 提出了一个专为流式视频设计的主动安全预警评测框架，核心贡献包括：

1. **数据集构建：** 740 段视频（481 风险 + 259 无风险），覆盖四个领域：驾驶、医疗、日常生活、工业生产。每段风险视频标注帧级别的**风险起始时刻**（risk onset）和**事故边界**（accident boundary）。

2. **严格评测指标：** 模型需在因果约束下（无未来帧）对流式视频进行观测，并输出时间校准的预警，同时要求内容正确性。

3. **基线评测：** 对 13 个主流视频 MLLM 进行系统性评测，发现在最严格指标下没有模型超过 20.0%，且召回率与误报率紧耦合。

**Story Arc:** "既有基准测视频'理解'→ 我们构建 PaSBench-Video，测视频'预见' "

*Existing benchmarks evaluate video understanding → we build PaSBench-Video to evaluate video foresight.*

---

## 创新性分析 / Innovation

1. **首个流式视频主动安全预警基准**：区别于已有安全评测（如 Video-SafetyBench），PaSBench-Video 强调**时序因果约束下的主动预警**，而非事后判定。
2. **帧级精细标注**：risk onset + accident boundary 双边界标注，支持对预警时机（是否"来得及"）的定量评估。
3. **四域覆盖**：驾驶/医疗/生活/工业，对工业内容安全监控具有直接参考价值。
4. **揭示了当前 MLLM 的核心瓶颈**：recall 与 FPR 的紧耦合问题——现有模型无法在保持低误报的同时维持高召回，暴露了视频安全监控商业化落地的主要障碍。

---

## 关键指标 / Key Metrics

| Dataset | Metric | Best Model | Score |
|---------|--------|------------|-------|
| PaSBench-Video (strict) | Strict Safety Warning Score | Seed-2.0-Pro (est.) | <20.0% |
| PaSBench-Video | Recall | All 13 models | Tightly coupled with FPR |
| PaSBench-Video (risk) | Risk Coverage | All tested | No model surpasses 20.0% on strictest metric |

**Baseline gap:** All 13 tested MLLMs remain below 20.0% on the strictest metric, indicating large room for improvement.

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 25 | First streaming proactive safety benchmark; dual-boundary annotation; rigorous causal constraint design |
| SOTA Delta | 15 | 12 | Comprehensive 13-model evaluation with new metrics; clear failure modes identified |
| Exp. Quality | 15 | 14 | 740 videos, 4 domains, frame-level annotations, 4 task types |
| Efficiency | 10 | 7 | Streaming evaluation design |
| Generalization | 5 | 5 | 4 diverse domains with wide applicability |
| Domain Relevance | 25 | 22 | Core relevance: video content safety monitoring directly applies to live-stream and short-video content governance |
| **Total** | **100** | **85** | |

---

## 与先前工作的对比 / Comparison with Prior Work

| Benchmark | Focus | Temporal Annotation | Streaming |
|-----------|-------|---------------------|-----------|
| Video-SafetyBench | Post-hoc safety classification | No | No |
| MVBench | General video understanding | No | No |
| ProactiveVideoQA | Proactive QA | No | No |
| **PaSBench-Video** | **Proactive safety warning** | **Frame-level dual boundary** | **Yes** |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

直播平台、短视频平台（如抖音、快手、淘宝直播）需要对流式视频内容进行实时安全预警，包括违规商品展示、危险行为演示等。PaSBench-Video 的评测体系可直接迁移到达人内容违规检测场景，评估 MLLM 在直播流中的主动预警能力。

Streaming video platforms (TikTok, Kuaishou, Taobao Live) require real-time safety alerts on live video content, including prohibited product demonstrations and dangerous behavior. PaSBench-Video's evaluation framework directly transfers to influencer content violation detection, assessing MLLM's proactive warning capability in live streams.

---

## Code Reproduction

See `code/PaSBench/` for a faithful PyTorch reproduction of the PaSBench-Video evaluation framework.
