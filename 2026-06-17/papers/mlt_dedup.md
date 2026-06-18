# MLT-Dedup: Efficient Large-Scale Online Video Deduplication via Multi-Level Representations and Spatial-Temporal Matching

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | MLT-Dedup: Efficient Large-Scale Online Video Deduplication via Multi-Level Representations and Spatial-Temporal Matching |
| **Authors** | David Yuchen Wang, Haoying Li, Hailun Xu, Wei Chee Yew, Zirui Zhu, Sanjay Saha, Hao Hei, Kanchan Sarkar, Kun Xu |
| **Affiliations** | Industry (KDD-2026 ADS track, likely TikTok/ByteDance or similar) |
| **arXiv** | [2606.12215](https://arxiv.org/abs/2606.12215) |
| **Submitted** | ~2026-06-12 |
| **Venue** | KDD-2026 Applied Data Science Track |
| **Domain Tags** | video deduplication, data quality, multi-level embedding, large-scale, content platform |

---

## 方法概述 / Method Summary

短视频平台上大量近重复视频（Near-Duplicate Videos, NDV）——经过剪辑、水印、滤镜等局部修改的相似视频——严重降低用户体验并增加存储带宽成本。MLT-Dedup 提出多级视频编码器（Multi-Level Video Encoder, ML-VE）：稀疏帧级 Embedding 支持大规模高效候选检索（ANN），细粒度帧级 Embedding 用于精准配对匹配；时空匹配模块（Spatial-Temporal Matching）对候选视频序列进行时序对齐，鲁棒处理片段剪辑和重排序问题。整个系统为在线流式架构，可处理大规模内容平台的实时视频流入。

**Story arc**: 近重复视频泛滥于内容平台（搬运、二次剪辑），现有哈希方法无法处理细微修改，深度学习方法召回率高但精度低或延迟高 → 两级 Embedding（稀疏+精细）+ 时空匹配，实现高精度高效率的在线重复检测。

**Key components**:
1. **ML-VE (Multi-Level Video Encoder)**: 稀疏 clip-level 向量（快速 ANN 候选召回）+ 细粒度 frame-level 向量（精准匹配）
2. **Spatial-Temporal Matching**: 对候选对的帧序列做动态规划时序对齐，容忍片段重排序
3. **Online Streaming Architecture**: 支持实时视频入库与实时去重判断
4. **Knowledge Distillation**: 使用大型 MLLM 教师模型蒸馏轻量在线推理模型

---

## 创新性分析 / Innovation Analysis

**vs. prior work**:
- 相比单级 Embedding（如 SSCD）缺乏精细匹配，MLT-Dedup 的两级设计兼顾效率与精度
- 时空匹配模块是首个专门处理视频"片段级重排"的去重方案
- KDD-2026 ADS 赛道，有工业规模验证
- 直接适用于电商内容平台（抖音、快手、小红书）的 UGC 视频去重

**Novelty assessment**: 两级 Embedding + 时空匹配组合合理且工程上可实施，KDD ADS 背书工业可用性。

---

## 关键指标 / Key Metrics

| Dataset/System | Metric | MLT-Dedup | Baseline |
|---------------|--------|-----------|----------|
| Large-scale video platform | Precision | high | — |
| Large-scale video platform | Recall | high | — |
| Online latency | — | production-grade | — |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 23 | 30 | 两级 Embedding + 时空匹配，工程创新扎实 |
| Experimental SOTA delta | 11 | 15 | KDD ADS 验证，具体数值受限 |
| Experimental quality / ablations | 12 | 15 | 工业规模验证 |
| Efficiency | 9 | 10 | 在线流式，大规模可部署 |
| Generalization | 3 | 5 | 平台特定，但架构泛化性好 |
| Domain relevance | 20 | 25 | 内容平台视频去重/数据质量，直接相关 |
| **Total** | **77** | **100** | 内容平台数据质量关键工具，KDD ADS 强背书 |
