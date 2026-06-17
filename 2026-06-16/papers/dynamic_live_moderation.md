# Dynamic Content Moderation in Livestreams

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching |
| **作者** | Wei Chee Yew, et al. (TikTok / ByteDance) |
| **机构** | ByteDance / TikTok |
| **链接** | https://arxiv.org/abs/2512.03553 |
| **arXiv ID** | 2512.03553 |
| **发表** | ACM SIGKDD 2026 (Proceedings of the 32nd ACM KDD, Jeju Island, Aug 2026) |
| **提交日期** | December 2025 (not in June 16 listing; included as KDD 2026 industry paper) |
| **Bucket** | STRONG |
| **Code** | `code/DynamicLiveModeration/` |

---

## 方法概述 / Method Overview

**中文：**  
该论文提出了一个双路径内容审核框架，专门针对直播流（live streaming）场景，由TikTok/ByteDance在生产环境中部署。系统包含两条并行管道：①**监督分类管道**（Supervised Classification Pipeline），针对已知的违规类别进行快速检测；②**相似度匹配管道**（Similarity Matching Pipeline），基于参考样本库对新型或细微违规进行识别。两条管道均利用多模态大语言模型（MLLM）进行知识蒸馏，将MLLM的理解能力注入轻量级模型，以保持在线推理的高效性。系统处理文本、音频、视觉三种模态的输入。

**English:**  
The paper presents a production-scale hybrid content moderation framework for livestreaming platforms (TikTok). A dual-pipeline architecture processes multimodal inputs (text, audio, visual): (1) a **supervised classification pipeline** for known violation types, and (2) a **reference-based similarity matching pipeline** for novel or subtle cases. Both pipelines are boosted by an MLLM that distills knowledge into compact models, keeping inference lightweight at scale.

---

## 故事弧线 / Story Arc

**现有方法不足 →** 传统分类器只能检测已知违规类别，对新型内容失效；纯MLLM推理成本极高。  
**本文设计 →** 双路径系统：分类器负责已知违规（低延迟），相似度匹配处理未知违规（基于MLLM蒸馏的嵌入检索），两者互补覆盖全谱。

---

## 创新性 / Innovation

1. **MLLM蒸馏策略**：将MLLM（高精度但高延迟）的能力蒸馏到两条轻量级管道，兼顾准确性与实时性。
2. **双路径互补设计**：分类路径处理已知违规，相似度路径处理未知/边缘违规，形成完整的违规检测覆盖。
3. **多模态融合**：文本+音频+视觉三模态联合处理，适合直播电商场景的复杂内容。
4. **工业级大规模验证**：在TikTok平台进行A/B测试，结果具有极高可信度。

**与前工作的差异：** 现有方法多为单模态或单管道；本文首次将MLLM蒸馏与多模态双路径系统相结合，并在生产环境验证。

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | 本文结果 | 对比 |
|-------------|------|----------|------|
| TikTok生产直播 | 分类管道 Recall@80%P | **67%** | 基线未公开 |
| TikTok生产直播 | 相似度管道 Recall@80%P | **76%** | > 分类管道 |
| TikTok A/B测试 | 用户观看违规直播减少 | **−6~8%** | 无基线（绝对值） |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 24 | 30 | 双路径MLLM蒸馏创新，工业价值极高 |
| 实验SOTA增量 (SOTA Delta) | 13 | 15 | 生产A/B测试结果可信 |
| 实验质量/消融 (Exp Quality) | 14 | 15 | 生产部署+对比实验+A/B测试 |
| 效率 (Efficiency) | 9 | 10 | 蒸馏保持轻量推理 |
| 泛化性 (Generalization) | 4 | 5 | 多模态，可迁移至电商平台 |
| 领域相关性 (Domain Relevance) | 24 | 25 | 直播电商内容审核，精准命中 |
| **Total** | **88** | **100** | |

---

## 复现代码位置

`code/DynamicLiveModeration/` — 完整复现代码（PyTorch实现）

---

## 参考链接

- arXiv: https://arxiv.org/abs/2512.03553  
- KDD 2026: https://dl.acm.org/doi/10.1145/3770854.3783936
