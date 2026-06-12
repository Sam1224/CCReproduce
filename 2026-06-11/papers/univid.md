# UNIVID: Unified Vision-Language Model for Video Moderation

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | UNIVID: Unified Vision-Language Model for Video Moderation |
| **Authors** | Kejuan Yang, Yizhuo Zhang, et al. (10 authors) |
| **Affiliation** | ByteDance |
| **ArXiv** | [2606.05748](https://arxiv.org/abs/2606.05748) |
| **Submitted** | June 4, 2026 |
| **Domain Tags** | `content-governance` `video-moderation` `VLM` `violation-detection` `live-streaming` |
| **Code** | `code/UNIVID/` (reproduction) |
| **Total** | **84 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 全球规模的视频内容审核面临双重挑战——需要细粒度多模态推理的同时，还需要可解释的输出。传统审核系统依赖分散的黑盒分类器，难以维护且缺乏透明度，无法应对平台上不断涌现的新型违规内容。

**解决方案：** UNIVID 提出以"策略感知字幕（policy-aware caption）"作为可解释的中间表示，构建统一的三阶段审核流水线，将粗粒度过滤、精细决策和趋势治理融为一体。

**Existing methods are insufficient → UNIVID solves it by:**  
现有方法（多个独立的黑盒分类器）→ UNIVID 通过生成策略感知字幕作为统一中间表示，使人工可验证，并支持多任务复用。

---

## 方法概述 / Method

UNIVID 是基于视觉语言模型（VLM）的视频审核系统，核心创新是用**策略感知字幕**（Policy-Aware Caption）作为可解释中间表示。整体流水线分三个级联阶段：

1. **Risk Filter（风险过滤器）**：多模态风险漏斗，将 UNIVID 生成的字幕与视觉特征融合，过滤潜在高风险视频。字幕提供语义对齐，视觉特征捕捉视觉信号。

2. **Moderation Actor（审核执行器）**：包含两个下游微调模型：
   - **UNIVID-Lite**：轻量化模型，预测审核决策（合规/违规分类）；
   - **UNIVID-RAG**：基于 RAG 的模型，通过检索已知违规事件召回漏审内容，提升召回率。

3. **Trend Governance（趋势治理模块）**：缓存 UNIVID 嵌入，通过微调特定趋势头（trend head）自适应检测新兴风险，无需重新训练完整模型。

训练数据：结合专家人工精标签与合成数据，解决现有 VLM 存在的安全护栏限制和策略对齐不足的问题。

---

## 创新性分析 / Innovation

**与现有工作的区别：**

| Aspect | Prior Work | UNIVID |
|--------|-----------|--------|
| 审核架构 | 多个独立黑盒分类器 | 统一 VLM 流水线 |
| 中间表示 | 无 / 不可解释特征 | 策略感知字幕（人工可验证）|
| 召回增强 | 固定规则 | UNIVID-RAG（基于历史违规事件）|
| 新兴风险 | 需重训完整模型 | 趋势头微调（高效适应）|
| 多任务复用 | 每个任务独立模型 | 共享 UNIVID 表示 |

**创新可信度：高。** 策略感知字幕方法实际解决了工业部署中的可解释性痛点，ByteDance 的生产背景使实验结果具有高可信度。

---

## 关键指标 / Key Metrics

| Dataset/Setting | Metric | UNIVID | Baseline |
|----------------|--------|--------|---------|
| Production platform | Recall@80%Precision | Significant improvement | N/A (production baseline) |
| Emerging risk detection | Detection latency | Adaptive (trend head) | Full retraining required |
| Multi-task | Shared representation | ✓ (policy captions) | ✗ |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 25 | 30 | Policy-aware caption as IIR is genuinely novel; three-stage pipeline addresses real production gaps |
| Experimental SOTA delta | 12 | 15 | Production deployment evidence from ByteDance platform |
| Experimental quality / ablations | 12 | 15 | Multi-stage evaluation; real-world production evidence |
| Efficiency | 7 | 10 | UNIVID-Lite for scale; trend head avoids full retraining |
| Generalization | 4 | 5 | Demonstrated across multiple violation categories |
| Domain relevance | 24 | 25 | Core video content moderation, violation detection, live-streaming governance — highest relevance |
| **Total** | **84** | **100** | |

---

## 中文摘要

UNIVID 是 ByteDance 提出的统一视觉语言模型视频审核系统。现有视频审核系统多采用多个分散的黑盒分类器，难以维护且缺乏透明度。UNIVID 提出以**策略感知字幕**为核心的统一中间表示，构建三阶段流水线：（A）风险过滤器融合多模态信号初步筛选高风险视频；（B）审核执行器包含 UNIVID-Lite（轻量审核决策）和 UNIVID-RAG（基于历史违规事件的召回增强）；（C）趋势治理模块通过缓存嵌入和微调趋势头，自适应检测新兴风险。该方法使审核决策可被人工验证，并支持多任务复用，显著降低模型维护成本。
