# Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching |
| **Authors** | Wei Chee Yew, Hailun Xu, Sanjay Saha, Xiaotian Fan, Hiok Hian Ong, David Yuchen Wang, Kanchan Sarkar, Zhenheng Yang, Danhui Guan |
| **Affiliation** | (Production platform team, industry) |
| **Venue** | KDD 2026 (32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Aug 9–13 2026, Jeju Island) |
| **arXiv** | https://arxiv.org/abs/2512.03553 |
| **Submitted** | December 2025 (v1) |
| **Code** | `code/HybridMod/` |

---

## 方法概述 / Method Summary

现有的直播内容审核系统通常依赖单一的有监督分类器，该分类器只能识别历史已知的违规类型，对新兴的、不断演化的违规内容（尤其是边缘案例）泛化能力差。

本文提出 **HybridMod**，一个混合直播内容审核框架，分两条并行流水线处理多模态（文本、音频、视觉）输入：

1. **有监督分类管道（Supervised Classification Pipeline）**：针对已知违规类别，训练专用多模态分类器。使用多模态大语言模型（MLLM）作为"教师"，通过知识蒸馏将 MLLM 对视频帧语义的理解迁移到轻量级分类器中，从而在推理阶段保持低延迟。
2. **相似性匹配管道（MLLM-Boosted Similarity Pipeline）**：针对新颖、模糊或难以定义规则的违规内容，以 MLLM 为核心构建参考库（reference gallery），通过跨模态相似性检索判断当前视频段与已标注违规样本的语义距离，若超过阈值则触发审核。

两条管道独立运行，各自输出结果，在最终决策层融合——保留分类器对常见违规的高精度，同时利用相似性管道捕获未见违规，形成互补。

**故事弧线：** 传统单一分类器只能检测已知违规类型，面对新兴内容束手无策 → HybridMod 以 MLLM 知识蒸馏驱动的双管道架构，分别应对已知和未知违规，实现 6–8% 的不良直播观看量下降。

---

## 创新性分析 / Innovation Analysis

- **新颖点1 — MLLM 双重作用**：MLLM 同时充当分类管道的蒸馏教师和相似性管道的特征提取器，避免在线推理时调用大模型，既保留了 MLLM 语义理解能力，又满足实时性需求。
- **新颖点2 — 双管道互补设计**：明确分离"已知违规"和"新兴违规"两类问题，两条管道专注各自任务，与单一端到端模型相比更易维护和迭代。
- **新颖点3 — 工业部署验证**：在生产环境进行大规模 A/B 测试，论文提供了真实业务指标（不良内容观看量减少），而非纯学术数据集评测。
- **与前作区别**：VideoModerator (EMNLP 2021) 只针对视觉模态，依赖离线标注；本文多模态处理（文本+音频+视觉），且明确针对"新兴违规"设计了参考相似性管道，并有线上 A/B 实验。

---

## 关键指标 / Key Metrics

| Dataset | Metric | This Work | Baseline |
|---------|--------|-----------|---------|
| Production (internal) | Recall @ 80% Precision (分类管道) | **67%** | — |
| Production (internal) | Recall @ 80% Precision (相似性管道) | **76%** | — |
| Online A/B Test | Reduction in unwanted livestream views | **6–8%** | — |

---

## 评分 / Score

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | 双管道 + MLLM 蒸馏架构新颖，但整体思路（分类+检索）在 CV 中有先例 |
| Experimental SOTA Delta | 12 | 15 | 生产环境 A/B 实验 6–8% 绝对提升，业务意义显著 |
| Experimental Quality / Ablations | 13 | 15 | KDD 2026 同行评审，生产规模验证，但消融细节在摘要层面未完全展开 |
| Efficiency | 8 | 10 | MLLM 知识蒸馏后推理无需大模型，轻量高效 |
| Generalization | 4 | 5 | 多模态输入，但数据集为内部数据，外部泛化能力待验证 |
| Domain Relevance (ecom + governance) | 25 | 25 | 直播电商内容审核，完美命中核心领域 |
| **Total** | **84** | **100** | |
