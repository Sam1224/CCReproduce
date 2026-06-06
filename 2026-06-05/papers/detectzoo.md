# DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| Title | DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities |
| arXiv | [2606.04205](https://arxiv.org/abs/2606.04205) |
| Submitted | June 2, 2026 |
| Authors | Sajad Ebrahimi et al. |
| Affiliation | Academic (multiple institutions) |
| Venue | arXiv preprint |
| Code | Available (PyPI: `detectzoo`) |
| Domain Tag | AIGC-detection · content-governance · multimodal · toolkit |

---

## 方法概述 / Method Summary

**English:**  
As generative AI proliferates, distinguishing human-created from AI-generated content (AIGC detection) is critical for content platforms, journalism, and e-commerce authenticity. Existing detectors are scattered across incompatible codebases with bespoke preprocessing and evaluation pipelines, making systematic comparison nearly impossible. DetectZoo unifies the AIGC detection ecosystem: it provides a single Python package with **61 detector implementations**, **22 benchmark dataset loaders**, and a **standardized evaluation pipeline** (multiple metrics, common interface). Each detector is self-contained, automatically fetches pretrained weights, and reproduces published results. DetectZoo covers text, image, and audio modalities.

**中文：**  
随着生成式 AI 普及，区分人类创作与 AI 生成内容（AIGC 检测）对内容平台的真实性维护至关重要。现有检测器分散在不兼容的代码库中，预处理和评估流程各异，系统性比较几乎不可能。DetectZoo 统一了 AIGC 检测生态：提供单一 Python 包，集成 **61 个检测器实现**、**22 个基准数据集加载器**和**标准化评估流水线**（多指标、统一接口）。每个检测器自包含，自动下载预训练权重，可复现已发布结果。覆盖文本、图像、音频三种模态。

---

## 故事弧线 / Story Arc

> **传统方案的不足 →** AIGC 检测研究碎片化，不同检测器代码不兼容，无法公平对比，复现成本极高。  
> **我们的方案 →** DetectZoo 构建统一工具箱，标准化整个实验流水线，让研究者一键接入 61 种检测器并公平比较。

---

## 关键指标 / Key Metrics

| Detector Count | Dataset Count | Modalities | Reproducibility |
|----------------|---------------|------------|----------------|
| 61 | 22 | Text, Image, Audio | Verified against published baselines |

---

## 评分 / Scoring

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation (max 30) | 12 | Primarily a software contribution; no new detection methodology |
| SOTA Delta (max 15) | 8 | Reproduces existing SOTA; no new performance improvements |
| Experimental Quality (max 15) | 10 | Comprehensive coverage of 61 detectors + 22 datasets |
| Efficiency (max 10) | 7 | Standard inference pipelines; automated weight fetching |
| Generalization (max 5) | 5 | Covers 3 modalities; extensible architecture |
| Domain Relevance (max 25) | 17 | Directly useful for AIGC detection in content governance; text+image cover key e-commerce cases |
| **Total** | **59** | |
