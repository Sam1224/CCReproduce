# Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| 字段 | 值 |
|------|-----|
| **Title** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **arXiv** | https://arxiv.org/abs/2605.17366 |
| **Authors** | (Authors not yet indexed; first-authored arXiv preprint) |
| **Affiliation** | (Not retrieved; likely industry/academia) |
| **Submitted** | May 17, 2026 |
| **Domain Tags** | `e-commerce` `multimodal-recommendation` `visual-representation` `item-retrieval` `VLM` |
| **Code** | Not provided |
| **Bucket** | STRONG |
| **Total** | **79** |

---

## 方法概述 / Method Overview

**中文：**
TGQ-Former（Text-Guided Q-Former）是一个文本引导的视觉表征学习框架，解决电商场景中商品图片常含促销覆盖层和背景杂乱（噪声视觉线索）导致检索鲁棒性下降的问题。框架采用**混合查询连接器（hybrid-query connector）**将视觉 token 流分为两个独立通道：一是以结构化商品元数据（品名、类目、属性）为锚点的元数据锚定流，二是保留自由探索的补充视觉流。进一步引入**轻量级可靠性感知双门控向量调制模块（reliability-aware dual-gated vector modulation）**，在噪声输入下自适应校准两路输出的贡献权重。

**English:**
TGQ-Former is a text-guided visual representation learning framework that tackles the problem of promotional overlays and background clutter in real-world product images degrading retrieval robustness in MLLM-style pipelines. A hybrid-query connector disentangles the visual token stream into two paths: a metadata-anchored path guided by structured item metadata, and a free exploratory path preserving complementary visual evidence. A lightweight reliability-aware dual-gated vector modulation module then adaptively calibrates the two paths' contributions under noisy inputs.

---

## 故事线 / Story Arc

MLLM 推荐管线中冻结的视觉编码器直接连接 LLM →  
商品图片的促销遮挡和背景杂乱注入噪声视觉线索 → 检索精度下降 →  
结构化元数据（商品文本）可作为语义锚，指导噪声鲁棒的视觉 token 抽取 →  
TGQ-Former 双路设计 + 门控调制 → 全库检索 HR@100 平均提升 6.04%。

---

## 创新性分析 / Innovation

- **问题定位精准**：识别了 MLLM 推荐管线中图像噪声与连接器设计的具体痛点（MLRM 风格的 Q-Former 缺乏语义引导）。
- **双路连接器**：将元数据引导的 token 与自由 token 显式解耦，比 naive Q-Former 更能抵抗图像噪声，是架构级别的真实改进。
- **门控调制**：可靠性感知门控是实现自适应噪声鲁棒性的关键，类似 Gating Network 但结合了视觉可靠性估计。
- **与先前工作差异**：传统 Q-Former（BLIP-2 等）不区分查询来源；TGQ-Former 明确区分元数据锚定查询和探索性查询，更符合工业电商场景。

---

## 关键指标 / Key Metrics

| Dataset | Metric | TGQ-Former | Baselines |
|---------|--------|-----------|-----------|
| Large-scale real-world e-commerce (full-pool retrieval) | Hit Rate@100 | **+6.04% avg** vs connector baselines | — |
| Same datasets | HR@100 | Consistently outperforms end-to-end MLLMs | — |

---

## 评分 / Scoring

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 21 / 30 | 双路 Q-Former + 门控调制是真实架构创新，但整体仍在 Q-Former 范式内延伸 |
| 实验指标 | 12 / 15 | 真实大规模电商数据全库检索 HR@100 平均 +6.04%，有实际意义 |
| 实验质量 | 11 / 15 | 真实数据集、全库检索，但论文细节（数据集名、消融细节）未完全获取 |
| 效率 | 7 / 10 | 轻量化连接器模块，较端到端 MLLM 效率更高 |
| 泛化性 | 4 / 5 | 全库检索设置接近工业实际场景 |
| 相关性 | 24 / 25 | 直接针对电商商品检索/推荐，噪声图片问题是达人带货场景的核心痛点 |
| **Total** | **79** | |
