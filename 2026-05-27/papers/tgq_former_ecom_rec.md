# Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **Authors** | Yufei Guo, Jing Ma, Tianlu Zhang, Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong, Jungong Han |
| **Affiliations** | Tsinghua University; JD.COM |
| **arXiv** | [2605.17366](https://arxiv.org/abs/2605.17366) |
| **Submitted** | 2026-05-17 |
| **Bucket** | STRONG |
| **Total** | **75 / 100** |

---

## 方法概述 / Method Overview

### EN
In real-world e-commerce, product images are noisy: they contain promotional text overlays, brand watermarks, and cluttered backgrounds that confuse visual encoders and degrade item-to-item (I2I) retrieval quality. Existing multimodal connectors (Q-Former, MLP, etc.) process visual tokens without considering the rich structured metadata (title, category, attributes) co-existing with the image. The proposed **Text-Guided Q-Former (TGQ-Former)** introduces *semantic guidance* into the visual token extraction process: structured metadata drives a **hybrid-query connector** that produces two complementary visual streams — (1) *metadata-anchored* tokens focused on content described by text, and (2) *exploratory* tokens capturing visual details not covered by metadata. A **reliability-aware dual-gated vector modulation** module then adaptively combines these streams, downweighting noisy channels when metadata-visual consistency is low. The framework is backbone-agnostic and plugs into existing multimodal recommendation architectures.

### ZH
电商真实场景中，商品图片含有促销文字叠层、品牌水印、杂乱背景，严重干扰视觉编码器，导致 Item-to-Item（I2I）检索精度下降。现有多模态连接器（Q-Former、MLP 等）在提取视觉 Token 时未利用商品的结构化元数据（标题、类目、属性）。本文提出 **Text-Guided Q-Former（TGQ-Former）**：结构化元数据驱动**混合查询连接器**，产生两路互补视觉流——(1) **元数据锚定 Token**（聚焦文字描述的内容）；(2) **探索性 Token**（捕捉文字未覆盖的视觉细节）。**可靠性感知双门控向量调制**模块自适应融合两路，在元数据-视觉一致性低时降低噪声通道权重。框架与主干无关，可插入现有多模态推荐架构。

---

## 故事弧 / Story Arc

> **"电商商品图片噪声导致多模态嵌入鲁棒性差"** → 用结构化元数据引导视觉 Token 提取，双流设计分离语义锚定与探索性特征，可靠性门控自适应降噪，在大规模真实电商数据上 H@100 提升 6.04%。

---

## 创新性分析 / Innovation Analysis

1. **元数据引导的视觉提取**：将文本元数据作为 cross-attention 查询端注入 Q-Former，而非独立处理。
2. **双流设计**：锚定流+探索流分工明确，避免单一特征同时承担对齐与发现两个目标的矛盾。
3. **可靠性感知门控**：动态评估元数据-视觉一致性，噪声商品图片时自动降权锚定流，设计优雅。
4. **工业级验证**：在真实电商全库检索（full-pool retrieval）上测试，而非小规模学术数据集。

---

## 关键指标 / Key Metrics

| Dataset | Metric | TGQ-Former | Baseline (best connector) |
|---------|--------|-----------|--------------------------|
| JD.COM E-commerce (large-scale real-world) | H@100 | **+6.04%** | baseline |

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 20 | 元数据引导 Q-Former + 双流 + 可靠性门控设计新颖 |
| Experimental SOTA delta | 15 | 10 | 6.04% H@100 提升在真实电商全库检索上意义显著 |
| Experimental quality / ablations | 15 | 12 | 真实大规模数据，有 ablation |
| Efficiency | 10 | 7 | Connector 增量设计，推理成本增加有限 |
| Generalization | 5 | 3 | 电商场景专用 |
| Domain relevance | 25 | 23 | 直接对应电商多模态商品检索和推荐 |
| **Total** | **100** | **75** | |

---

## 电商治理价值 / E-commerce Governance Value

- **商品图片噪声处理**：解决卖家上传促销图、水印图导致检索失准的核心问题
- **商品 I2I 召回**：提升"猜你喜欢"、相似商品推荐质量
- **多模态商品理解**：元数据-图像协同表示可用于达人商品推广内容的商品侧验证
