# Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching |
| **arXiv ID** | 2512.03553 |
| **提交日期** | December 2025 (published at KDD 2026) |
| **作者** | (ByteDance / Douyin 内容安全团队，KDD 2026) |
| **机构** | Industry (live e-commerce platform) |
| **论文链接** | https://arxiv.org/abs/2512.03553 |
| **代码** | code/LivestreamMod/ |
| **venue** | KDD 2026 |
| **桶** | STRONG |
| **Total** | **85** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 现有直播内容审核依赖**单一**的有监督分类器，面对"策略边界模糊"与"不断涌现的新型违规"时泛化能力不足。本文提出**双路径混合架构**：① 监督多类分类器快速处理已知违规；② 基于 MLLM 知识蒸馏的参考相似度检索器处理模糊 / 新兴违规，二者联合投票。

核心技术：
1. **双路径（Dual-Path）**：分类路径（Supervised Classification）+ 相似度路径（Reference-based Similarity Retrieval），两路均消费 text / audio / visual 三模态输入。
2. **MLLM 教师蒸馏**：用大型多模态模型（MLLM）作为教师，对轻量分类器和轻量 re-ranking 模型进行知识蒸馏，使生产推理成本可控。
3. **对比预训练（CLIP-loss）**：相似度路径使用 CLIP-loss 进行跨模态对比预训练，使 text/audio/visual 特征对齐到同一语义空间。
4. **联合决策**：分类路径覆盖高置信度已知类别，相似度路径通过检索"参考违规案例库"捕获新兴违规，hybrid 结合后 F1 大幅超越单路。

**与前工作的差异：**
- 前工作：单模态或多模态分类器（需要大量标注、对新类泛化差）。
- 本文：引入 retrieval 路径，无需新类型标注即可泛化；MLLM 蒸馏让大模型知识赋能小模型线上推理。

---

## 关键指标 / Key Metrics

| 指标 | 分类路径 | 相似度路径 | 混合 |
|------|----------|----------|------|
| Recall @ 80% Precision | 67% | 76% | 显著 > 单路 |
| 模态 | text + audio + visual | text + audio + visual | — |
| 部署规模 | Production-scale 直播平台 | — | — |

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 22 | 双路 MLLM-蒸馏混合架构；检索路径弥补分类路径弱点 |
| SOTA Δ (max 15) | 13 | 生产部署验证，76%recall@80%precision；hybrid 明显优于单路 |
| Experimental Quality (max 15) | 13 | 生产级数据、三模态、两路 ablation |
| Efficiency (max 10) | 8 | 教师蒸馏保持在线推理轻量 |
| Generalization (max 5) | 4 | 多模态 + 多平台可迁移 |
| Domain Relevance (max 25) | 25 | **直播内容审核**，电商达人治理核心场景 |
| **Total** | **85** | — |

---

## 创新性分析

1. **检索路径解决冷启动**：新型违规无需重新训练分类器，只需向案例库补充参考样本即可在相似度路径生效——这是策略快速迭代的关键能力。
2. **MLLM 蒸馏**：将 7B+ 多模态教师的语义理解压缩至轻量模型，实现生产可用推理延迟。
3. **三模态对齐**：text/audio/visual 共同参与——直播场景中语音（主播话术）和视觉（背景/展示品）同等重要，单模态审核易被规避。

---

## 电商 / 达人治理落地思路

- 分类路径 → 处理高频已知违规（虚假宣传、违禁品展示）
- 相似度路径 → 处理新出现的违规套路（规避关键词、隐晦手势）
- 跨场次检索 → 同一达人历史高风险场次记录作为参考，支持"惯犯检测"
