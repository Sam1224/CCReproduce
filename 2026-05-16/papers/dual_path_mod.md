# Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching |
| **Authors** | (Production team — names from ACM DL proceedings) |
| **Affiliation** | Large-scale video streaming platform (industrial) |
| **arXiv ID** | [2512.03553](https://arxiv.org/abs/2512.03553) |
| **Venue** | ACM SIGKDD 2026 (32nd), Applied Data Science Track |
| **Submission Date** | December 2025 (KDD 2026 accepted) |
| **Domain Tags** | `#content-moderation` `#live-streaming` `#MLLM` `#similarity-matching` `#knowledge-distillation` `#production` |
| **Code** | See `code/DualPathMod/` in this repo |
| **Total** | **82 / 100** |

---

## 故事弧线 / Story Arc

**现有问题 (Problem):** 直播平台内容审核面临两个根本性矛盾：(1) **已知违规** — 精确但静态：监督分类模型对训练见过的违规类别精确有效，但对新型违规行为泛化性差；(2) **相似度匹配** — 泛化但低精度：基于语义嵌入的相似检索可发现新型违规，但误报率高、计算成本大。此外，大型多模态语言模型（MLLM）虽理解力强，但延迟过高，无法用于实时直播审核。

**设计方案 (Solution):** 提出**双路径混合审核架构（Dual-Path Hybrid Moderation）**：路径一用监督分类器处理显式高置信度违规；路径二用语义嵌入+近邻检索处理新型/隐式违规。两条路径均通过 MLLM 进行知识蒸馏，将 MLLM 的理解能力注入轻量级模型，在生产环境中实现高效部署。

---

## 方法概述 / Method Overview

### 系统架构 (System Architecture)

```
Input: Live Video Stream (multimodal: visual + audio + text/subtitles)
         │
         ▼
   ┌─────────────────────────────────────────────────┐
   │           Feature Extraction                    │
   │  Video Encoder + Audio Encoder + Text Encoder   │
   └──────────┬─────────────────────────┬────────────┘
              │                         │
              ▼                         ▼
   ┌──────────────────┐      ┌─────────────────────────┐
   │  Path 1:         │      │  Path 2:                │
   │  Supervised      │      │  Similarity Matching    │
   │  Classification  │      │  (Semantic + Perceptual │
   │                  │      │   Embedding + kNN)      │
   │  ┌────────────┐  │      │  ┌───────────────────┐  │
   │  │ MLLM       │  │      │  │ MLLM-boosted      │  │
   │  │ Distill    │  │      │  │ Embedding         │  │
   │  └────────────┘  │      │  │ Distillation      │  │
   └────────┬─────────┘      └──────────┬────────────┘
            │                           │
            └────────────┬──────────────┘
                         ▼
                   Decision Fusion
                  (OR / Score-based)
                         │
                         ▼
              Moderation Decision (PASS / REMOVE / REVIEW)
```

### 关键技术细节

**Path 1 — Supervised Classification:**
- 多模态特征联合训练的分类头
- MLLM 通过软标签蒸馏提升细粒度违规区分能力
- 在线低延迟推理（MLLM 仅用于离线蒸馏，不在线运行）

**Path 2 — MLLM-Boosted Similarity Matching:**
- 构建"已知违规案例库"（Reference Policy Violation Store）
- MLLM 为每个参考样本生成高质量语义嵌入描述
- 新内容实时计算嵌入，与违规案例库做近邻检索
- 语义嵌入 + 感知嵌入双通道，覆盖语义违规和视觉违规

**MLLM Knowledge Distillation:**
- Teacher: Large MLLM（理解力强，延迟高）
- Student: 轻量级多模态编码器（实时可用）
- 蒸馏目标：(a) 分类软标签；(b) 嵌入空间对齐

---

## 关键指标 / Key Metrics

| Metric | Value | Baseline | Context |
|--------|-------|----------|---------|
| Classification path Recall @ 80% Precision | **67%** | N/A (prior prod) | Production deployment |
| Similarity path Recall @ 80% Precision | **76%** | N/A | Production deployment |
| Online A/B: Reduction in unwanted livestream views | **6–8%** | 0% | Large-scale A/B test |

> The similarity matching path (76%) outperforms the classification path (67%) at equal precision, demonstrating the value of the MLLM-boosted approach for novel violation discovery.

---

## 创新性分析 / Innovation Analysis

**vs. Prior Work:**
- 传统纯分类模型：对已知违规有效，但对新型违规泛化差
- 传统纯相似检索：泛化好但误报率高
- 本文首次在直播场景实现：双路径统一框架 + MLLM 知识蒸馏 + 生产级 A/B 验证

**可行性评估:** 工业级系统，已在大规模平台部署。双路径互补设计直觉合理，MLLM 蒸馏是成熟方案。最大创新在于将蒸馏后的 MLLM 能力注入实时相似性检索，实现对"未见违规"的泛化。

**局限性:** 相似路径依赖高质量违规案例库维护；当新违规与库中样本语义差异大时效果有限。

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | 双路径 MLLM 蒸馏架构新颖；将 MLLM 能力注入实时检索是独特贡献 |
| Experimental SOTA Delta | 12 | 15 | 生产 A/B 6-8% 减少；76% 召回@80%精确 — 实际影响显著 |
| Experimental Quality | 12 | 15 | 生产级 A/B 测试，大规模验证；KDD 双盲评审通过 |
| Efficiency | 8 | 10 | MLLM 仅离线蒸馏；Student 模型实时运行 |
| Generalization | 4 | 5 | 相似路径可处理未见违规类型 |
| Domain Relevance | 24 | 25 | 直播内容治理核心场景；高度匹配达人/直播平台违规检测 |
| **Total** | **82** | **100** | |

---

## 与我方业务的关联 / Relevance to Business

- **直播违规检测**: 双路径架构可直接用于直播带货违规实时检测
- **达人内容治理**: 相似性检索可用于发现"打擦边球"的新型违规内容
- **知识蒸馏部署**: MLLM 蒸馏方案解决线上推理延迟问题
- **冷启动**: 违规案例库启动成本低，适合新政策快速上线

**代码复现路径:** `code/DualPathMod/`
