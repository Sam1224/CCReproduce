# Valley3: Scaling Omni Foundation Models for E-commerce

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Valley3: Scaling Omni Foundation Models for E-commerce |
| **Authors** | Valley3 Team (ByteDance / Lark AI) |
| **Affiliation** | ByteDance Inc. |
| **arXiv ID** | [2605.01278](https://arxiv.org/abs/2605.01278) |
| **Submission Date** | May 2026 |
| **Domain Tags** | `#e-commerce` `#MLLM` `#omni-multimodal` `#content-governance` `#live-streaming` `#product-understanding` |
| **Code** | See `code/Valley3/` in this repo |
| **Total** | **84 / 100** |

---

## 故事弧线 / Story Arc

**现有问题 (Problem):** 现有通用多模态大语言模型（MLLM）在处理电商场景时面临多重挑战：(1) 缺乏对电商专属多模态信号（商品图、短视频带货、直播、多语言音频）的深度理解；(2) 对电商业务推理（营销逻辑、消费者意图分析、合规判断）能力不足；(3) 复杂场景下推理效率与精度难以平衡；(4) 现有电商模型不具备主动搜索与深度研究的能力。

**设计方案 (Solution):** Valley3 是一个 Omni 多模态大语言模型，通过**四阶段电商持续预训练流水线**，依次获得音频理解、跨模态指令跟随、电商领域知识和长上下文推理能力；再通过后训练引入**可控推理模式**（1个无思考模式+3个不同深度思考模式）；并配备**主动搜索（Agentic Search）**能力，支持电商深度研究任务。

---

## 方法概述 / Method Overview

### 四阶段 Omni 电商预训练 (Four-Stage Omni E-commerce Pre-training)

```
Stage 1: Audio Understanding Pre-training
   ├── Multilingual audio encoder alignment
   └── Audio-visual cross-modal matching in short-video context

Stage 2: Cross-Modal Instruction Following
   ├── Text / Image / Video / Audio joint instruction tuning
   └── Long-context omni data

Stage 3: E-commerce Domain Knowledge
   ├── Product understanding: title, attribute, category
   ├── Livestream content analysis: host, product demo, audience
   └── Moderation & Governance: violation detection, policy adherence

Stage 4: Long-Context Reasoning
   ├── Multi-turn e-commerce dialogue
   └── Deep research: agentic retrieval + synthesis
```

### 可控推理模式 (Controllable Reasoning)
- **Non-thinking mode**: fast response for simple classification/retrieval tasks
- **Level-1/2/3 thinking**: progressively deeper chain-of-thought for complex e-commerce reasoning (violation analysis, consumer insight)

### Agentic Search
- 主动调用搜索工具（商品数据库、政策库、用户评论库）
- 支持电商深度研究任务（竞品分析、选品决策）

### 原生多语言音频 (Native Multilingual Audio)
- 首个为电商短视频场景设计的原生音频-视觉多模态模型
- 支持中英日韩等多语言直播解说

---

## 关键指标 / Key Metrics

| Dataset / Benchmark | Metric | Valley3 | Baseline (Best) | Δ |
|--------------------|--------|---------|-----------------|---|
| In-house E-com (Product Understanding) | Accuracy | — | — | +7%+ avg |
| In-house E-com (Livestream Analysis) | Accuracy | — | — | +7%+ avg |
| In-house E-com (Moderation & Governance) | F1 | — | — | +7%+ avg |
| Open-source General Benchmarks | Various | Competitive | Strong MLLM baselines | ≈0 regression |

> Note: Exact numbers are from the paper's abstract description. Full tables require accessing the paper PDF directly.

---

## 创新性分析 / Innovation Analysis

**vs. Prior Work:**
- Valley (v1/v2): 早期版本仅支持图文，Valley3 扩展至 Omni（文本/图像/视频/音频）
- Qwen-VL / InternVL / LLaVA: 通用 VLM，不针对电商场景的四阶段持续预训练
- 主要创新：(1) 第一个原生支持多语言音频的电商 MLLM；(2) 可控推理模式实现推理效率与深度的灵活平衡；(3) Agentic Search 扩展电商深度研究能力

**可行性评估:** 创新路径清晰，四阶段预训练是实用工程路线；可控推理参考了思维链研究；Agentic Search 是工业界成熟方案。整体创新幅度中等偏高，工程贡献显著。

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 26 | 30 | 首个原生音频+可控推理的电商 Omni MLLM；四阶段设计系统新颖 |
| Experimental SOTA Delta | 11 | 15 | 内外部基准均 7%+ 提升；具体数值受限于摘要 |
| Experimental Quality | 11 | 15 | 内部+开源双轨评测，覆盖电商三大核心任务 |
| Efficiency | 7 | 10 | 可控推理模式有效平衡推理成本 |
| Generalization | 4 | 5 | 通用基准竞争力保持良好 |
| Domain Relevance | 25 | 25 | 完全针对电商生态：商品理解、直播分析、合规治理 |
| **Total** | **84** | **100** | |

---

## 与我方业务的关联 / Relevance to Business

- **商品理解 (Product Understanding)**: 直接对标商品标题、属性、类目理解任务
- **内容合规 (Content Moderation & Governance)**: 内置违规检测、政策遵从判断能力
- **直播分析 (Livestream Analysis)**: 直播带货场景的多模态理解（主播、商品展示、弹幕）
- **达人治理 (Influencer Governance)**: 可用于评估达人内容质量与合规性
- **深度研究 (Deep Research)**: Agentic Search 可用于选品决策、竞品分析

**代码复现路径:** `code/Valley3/`
