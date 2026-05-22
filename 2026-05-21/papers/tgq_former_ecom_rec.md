# TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| 作者 | (from arXiv 2605.17366 — full author list not retrieved due to 403; industrial e-commerce affiliation inferred) |
| 机构 | Industrial e-commerce research (affiliation pending direct access) |
| arXiv | https://arxiv.org/abs/2605.17366 |
| 提交日期 | 2026-05-17 (v1); likely in May 21 arXiv listing |
| 领域标签 | 电商推荐 · 多模态嵌入 · VLM · Item-to-Item Retrieval · 噪声鲁棒性 |
| 桶类别 | STRONG |
| **总分 / Total** | **81 / 100** |

---

## 方法概述 (中文)

现有电商 Item-to-Item (I2I) 检索模型依赖 MLLM 对商品图文进行统一嵌入，但真实场景中商品图片常含促销遮罩、背景杂乱等视觉噪声，导致检索召回不稳定。简单的 Q-Former 连接器等方法倾向于将这些噪声视觉线索与商品语义一并编码，严重影响相似性度量准确度。

**TGQ-Former** 提出文本引导的视觉表示学习框架，核心设计如下：
1. **混合查询连接器 (Hybrid-Query Connector)**：将 Q-Former 中的查询分为两路——元数据锚定流 (Metadata-Anchored Stream) 与探索视觉流 (Exploratory Visual Stream)。前者以商品结构化文本（类目、品牌、属性）为语义引导，通过交叉注意力聚焦与商品真实属性相关的图像区域；后者保留自由探索，捕捉文本未覆盖的细粒度视觉证据。
2. **可靠性感知双门控向量调制 (Reliability-Aware Dual-Gated Vector Modulation, RDGVM)**：在噪声图像下动态调控两路视觉流的贡献权重。当检测到图像可靠性下降（如大面积遮罩），模块自动降低探索流权重，提升元数据引导流的主导性，从而实现噪声鲁棒性。

整个框架在冻结的视觉编码器基础上只引入轻量级连接器（Q-Former + RDGVM），适合工业部署。

---

## 故事线 (Story Arc)

> **现状不足：** 电商 I2I 检索中，商品图像普遍含有促销贴纸/水印/背景噪声，现有 MLLM 连接器（Q-Former、MLP Connector 等）无法区分噪声视觉线索与真实商品特征，导致 Hit Rate@100 在噪声场景下显著下降。
>
> **我们的解法：** TGQ-Former 用商品结构化元数据作为语义锚点引导视觉 token 提取，同时引入双门控模块在噪声下自适应调权——以文本驯化视觉，以可靠性调制融合。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 混合查询 Q-Former + 可靠性感知双门控调制：首次将商品结构化文本作为 Q-Former 的语义约束，明确区分元数据引导视觉与自由视觉探索两路 |
| vs. 先前工作 | 传统 Q-Former (InstructBLIP) / MLP connector 统一嵌入不区分噪声源；MAPS/ECLIP 等方法缺乏噪声鲁棒机制；端到端 MLLM 推理效率低不适合工业检索 |
| 可行性 | 在大规模真实电商数据集上验证，全量检索池评测，工业可信度高 |
| 局限 | 仅在单一平台数据验证；结构化文本依赖标注质量；对极端遮罩（如全图贴图）的鲁棒性未充分讨论 |

---

## 关键指标

| 数据集 | 指标 | TGQ-Former | 最强基线 |
|--------|------|-----------|---------|
| 大规模工业电商数据集（全量检索池） | Hit Rate@100 (H@100) | +6.04% avg | connector baselines |
| 大规模工业电商数据集 | H@100 (含噪声图像子集) | 显著优于端到端 MLLM | end-to-end MLLM |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 24 | 30 | 混合 Q-Former + 双门控调制设计新颖；元数据引导视觉的思路具有独立价值 |
| 实验 SOTA Delta | 11 | 15 | H@100 +6.04% 在工业全量检索池上是有意义的提升 |
| 实验质量/消融 | 12 | 15 | 工业规模数据；全量检索（非采样）评测；有多组 connector baseline 对比 |
| 效率 | 8 | 10 | 轻量连接器，适合工业 I2I 检索部署 |
| 泛化性 | 3 | 5 | 仅限单平台电商；未验证跨品类 |
| 领域相关性 | 23 | 25 | 直接对应电商商品多模态检索/推荐核心场景 |
| **总分 / Total** | **81** | **100** | — |

---

## 代码与数据

- 无公开代码仓库
- 数据集：工业电商内部数据，未公开
- **本报告完整 PyTorch 复现：** `code/TGQFormer/`
