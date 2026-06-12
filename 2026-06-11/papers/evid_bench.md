# EVID-Bench: When Seeing Is Not Believing — A Benchmark for Search-Grounded Video Misinformation Detection

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | When Seeing Is Not Believing -- A Benchmark for Search-Grounded Video Misinformation Detection |
| **Authors** | (Multi-institution) |
| **Affiliation** | (Multi-institution) |
| **ArXiv** | [2606.04098](https://arxiv.org/abs/2606.04098) |
| **Submitted** | June 3, 2026 |
| **Domain Tags** | `content-governance` `misinformation-detection` `video-benchmark` `multimodal` `MLLM` |
| **Total** | **70 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 视频虚假信息越来越多地通过语义和证据层面操作（选择性剪辑、时序重排、跨源拼接、AI 内容增强），仅凭输入视频本身无法可靠验证，因为缺失、重排、替换或重新情境化的证据位于视频之外。

**解决方案：** 提出 EVID-Bench，一个**基于搜索的视频虚假信息检测**基准，系统必须在开放网络中搜索相关视频，并通过跨视频比较识别虚假信息。

**Story arc:** Single-video inspection is insufficient → Cross-video search-grounded verification is required.

---

## 方法概述 / Method

**EVID-Bench 基准设计：**

1. **数据规模：** 222 个视频样本
2. **操纵类型分类（9种，3大类）：**
   - **AI 生成类（AI Generation）**：完全由 AI 生成的虚假视频
   - **单源剪辑类（Single-Source Editing）**：对原始视频进行选择性剪辑、时序重排等操作
   - **多源拼接类（Multi-Source Editing）**：从多个视频源拼接，混淆真实场景
3. **验证标准：** 所有样本经过人工验证，确认前沿模型仅通过视觉检查无法检测（单视频方法失效）
4. **评估范式：** 系统需调用网络搜索 API 检索相关视频，并进行跨视频语义比对

**核心洞察：** 当前最先进的多模态大模型（前沿 MLLMs）在仅基于视频本身的检测上表现接近随机，必须借助外部搜索才能有效检测。

---

## 创新性分析 / Innovation

**与现有基准的区别：**

| Aspect | Existing Benchmarks | EVID-Bench |
|--------|-------------------|------------|
| 检测范式 | 单视频检测 | 搜索增强的跨视频比对 |
| 操纵类型 | 主要是 deepfake | 9种操纵类型含多源拼接 |
| 样本验证 | 自动标注 | 人工验证（前沿模型单视频检测失效）|
| 信息依赖 | 视频内部特征 | 开放网络证据搜索 |

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| Frontier MLLMs (visual only) | Detection accuracy | ≈ random (all fail) |
| EVID-Bench total | Samples | 222 videos |
| Manipulation types | Coverage | 9 types, 3 categories |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | Search-grounded paradigm is novel; demonstrates fundamental limitation of visual-only inspection |
| Experimental SOTA delta | 10 | 15 | Shows frontier models fail, motivating the benchmark |
| Experimental quality / ablations | 11 | 15 | Manual verification of all 222 samples is rigorous |
| Efficiency | 5 | 10 | Not efficiency focused |
| Generalization | 4 | 5 | 9 manipulation types provide broad coverage |
| Domain relevance | 20 | 25 | Content governance, fake content detection for platforms |
| **Total** | **70** | **100** | |

---

## 中文摘要

本文提出 EVID-Bench，一个**基于网络搜索的视频虚假信息检测**基准。随着 AI 生成和编辑技术的发展，视频虚假信息越来越多地通过选择性剪辑、时序重排、跨源拼接等语义层面的操作实现，仅凭单视频视觉检查无法识别。EVID-Bench 包含 222 个视频，涵盖 3 大类 9 种操纵类型（AI 生成、单源剪辑、多源拼接），所有样本均经过人工验证确认前沿 MLLM 无法通过单视频检测发现。基准要求系统通过开放网络搜索相关视频并进行跨视频比对来识别虚假信息，为内容安全团队提供了全新的评估范式。
