# TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **Title** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **Authors** | (Full list not recovered from WebSearch; affiliation inferred: Chinese e-commerce/industry lab) |
| **Affiliation** | Industry lab (details in full paper) |
| **arXiv** | [2605.17366](https://arxiv.org/abs/2605.17366) |
| **Submitted** | ~16–17 May 2026 |
| **Domain** | E-Commerce Recommendation · Multimodal Retrieval · VLM Connector |
| **Bucket** | STRONG |
| **Code** | No official code released |

---

## 分数 / Score

| 维度 | 分数 | 满分 |
|---|---|---|
| Innovation | 23 | 30 |
| Experimental SOTA delta | 11 | 15 |
| Experimental quality / ablations | 12 | 15 |
| Efficiency | 6 | 10 |
| Generalization | 4 | 5 |
| Domain relevance (ecom + governance) | 23 | 25 |
| **Total** | **79** | **100** |

**Justification**: High innovation for introducing text metadata as semantic guidance for visual token extraction in Q-Former-style connectors — addressing a real pain point of noisy product images. The reliability-aware dual-gated modulation is a practical engineering contribution. 6.04% H@100 improvement on large-scale real e-commerce datasets is solid. Domain relevance is very high: product image retrieval with noisy overlays is a direct day-to-day challenge in e-commerce platforms.

---

## 方法概述 / Method Summary

电商场景下的商品图像往往包含促销贴图、水印和背景杂乱等噪声，直接降低了多模态大语言模型（MLLM）的视觉检索鲁棒性。现有的 MLRM 风格流水线使用冻结视觉编码器 + 轻量连接器（如 Q-Former），但连接器只能被动聚合视觉 token，无法利用商品结构化元数据对视觉信号进行语义引导，导致对噪声极为敏感。

**TGQ-Former (Text-Guided Q-Former)** 的核心设计：

1. **文本引导的双路视觉流 (Hybrid-Query Connector)**  
   - **元数据锚定流 (Metadata-Anchored Stream)**: 将商品标题、类目、属性等结构化文本作为 query，通过 cross-attention 从视觉编码器中提取语义对齐的视觉 token。  
   - **探索性视觉流 (Exploratory Visual Stream)**: 保留传统 Q-Former 的可学习 query，捕捉文本元数据未能覆盖的视觉细节（如颜色纹理、细粒度款式）。

2. **可靠性感知双门调制模块 (Reliability-Aware Dual-Gated Vector Modulation)**  
   - 在推理时动态评估两个流的可靠性（基于噪声程度），自适应地调整各流的贡献权重。  
   - 当图像噪声较高时，更多依赖文本引导流；图像清晰时，平衡两路信号。

3. **训练目标**: 商品检索对比损失（InfoNCE），在商品图文对上进行全库检索评估。

**Story Arc**: 商品图像噪声使得 Q-Former 连接器学到错误的视觉语义 → TGQ-Former 通过结构化元数据引导视觉 token 提取 + 双门可靠性调制，实现噪声鲁棒的多模态电商检索。

---

## 创新性分析 / Innovation Analysis

**与 prior work 的对比**:
- 标准 Q-Former（BLIP-2）：可学习 query 无约束，在噪声图像上容易捕捉到无效视觉特征。
- CLIP/BLIP 直接用于电商检索：未利用结构化商品属性。
- TGQ-Former 的独特性：  
  (a) **首次** 将商品结构化元数据（文本）作为 Q-Former query 的语义锚，而非仅用于 caption 生成；  
  (b) **双路设计** 平衡语义约束与视觉发现；  
  (c) **自适应门控** 根据图像质量动态调整权重——工程实用性强。

**可行性**: 基于成熟的 Q-Former 框架改造，额外参数量小，适合工业部署。

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | TGQ-Former | Best Baseline | 提升 |
|---|---|---|---|---|
| 大规模电商数据集（真实全库检索）| Hit Rate@100 (H@100) | SOTA | 连接器基线 | +6.04% avg |
| 真实电商数据集 | Recall@10 | 领先 | end-to-end MLLM | 显著提升 |

*具体数值以论文正文为准；上述为论文摘要/描述中报告的核心结论。*

---

## 与电商内容生态的关联

- **商品搜索与召回**: TGQ-Former 的全库检索设置（full-pool retrieval）直接对应电商搜索召回阶段。H@100 的提升意味着更多相关商品能在 top-100 内被召回，支撑精排模块。
- **达人营销内容匹配**: 当达人发布内容时，平台需要快速匹配相关商品。TGQ-Former 的图文双路设计可用于"内容→商品"跨模态检索。
- **数据质量鲁棒性**: 噪声图像在电商平台中普遍存在（商家自上传图片质量参差不齐），TGQ-Former 的鲁棒性设计对真实数据尤为重要。

---

## 论文链接

- arXiv: https://arxiv.org/abs/2605.17366
