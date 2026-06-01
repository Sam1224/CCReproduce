# TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **arXiv ID** | 2605.17366 |
| **提交日期** | ~2026-05-17（2605 序列估计） |
| **作者** | (未完整披露，来自工业界 e-commerce) |
| **机构** | E-commerce platform research |
| **论文链接** | https://arxiv.org/abs/2605.17366 |
| **桶** | STRONG |
| **Total** | **75** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 工业电商图片（商品主图）中存在大量**促销贴纸、文字叠加、背景杂乱**等"噪声"，当 VLM/MLLM 被用于 item-to-item (I2I) 检索时，视觉编码器会将这些无关噪声编码进嵌入向量，导致检索质量下降。本文提出 **TGQ-Former**（Text-Guided Q-Former），以商品结构化元数据（标题、品类、属性）为语义引导，"蒸馏"出干净的视觉 token，同时保留补充视觉信息，通过可靠性感知的双门控向量调制（Dual-Gated Vector Modulation, DGVM）自适应融合两路视觉流。

**架构：**
```
输入: 商品图片 + 结构化元数据（标题/属性/品类）
         ↓
[Frozen Vision Encoder]  [Text Encoder (元数据)]
         ↓                       ↓
[TGQ-Former Hybrid-Query Connector]
   ├─ Query Stream A (metadata-anchored):  元数据引导的语义 token 提取
   └─ Query Stream B (exploratory):        自由探索视觉 token
         ↓
[DGVM: Reliability-Aware Dual-Gated Vector Modulation]
   → 动态校准两路贡献（噪声高时降低 B 的权重）
         ↓
  最终多模态商品嵌入 → I2I 检索
```

**与前工作差异：**
- 传统 Q-Former：单路 query，无文本引导，噪声无过滤
- TGQ-Former：双路 + 文本引导 + 可靠性门控 = 对噪声图片鲁棒

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | TGQ-Former | 最佳 Baseline |
|--------|------|-----------|--------------|
| 大规模真实电商 I2I (full-pool) | Hit Rate@100 (H@100) | +6.04%（平均） | SOTA connector baseline |

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 20 | 双路 hybrid Q-Former + DGVM 设计新颖，专攻电商噪声图片 |
| SOTA Δ (max 15) | 11 | +6.04% H@100 在大规模真实数据上 |
| Experimental Quality (max 15) | 12 | 全池检索（full-pool）评测最严格；真实工业数据 |
| Efficiency (max 10) | 6 | 轻量 connector，不训练视觉编码器 |
| Generalization (max 5) | 3 | 电商专向，但可迁移至其他含噪图片检索 |
| Domain Relevance (max 25) | 23 | 直接解决电商图片 I2I 检索核心痛点 |
| **Total** | **75** | — |

---

## 创新性分析

1. **元数据引导视觉提取**：用商品文本信息作为视觉 token 提取的"锚点"，确保提取到的视觉特征与商品语义对齐而非被噪声主导。
2. **DGVM 可靠性门控**：当输入图片噪声高（元数据覆盖度低）时，自动降低 exploratory stream 的贡献——类似注意力机制的自适应去噪。
3. **Frozen Encoder + Light Connector**：不微调视觉编码器，保持通用性；只训练轻量连接器，工业部署成本低。

---

## 电商 / 达人治理落地思路

- 商品 I2I 检索：用于相似商品推荐、重复商品去重
- 内容质量：噪声过滤后的视觉嵌入可用于内容质量评分
- 达人选品：基于干净商品嵌入的语义检索，支持达人选品匹配
