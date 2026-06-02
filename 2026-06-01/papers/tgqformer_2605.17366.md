# Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation (TGQ-Former)

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **Authors** | Yufei Guo, Jing Ma, Tianlu Zhang (Tsinghua University); Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong (JD.COM); Jungong Han (Tsinghua University) |
| **Affiliation** | Tsinghua University + JD.COM |
| **Venue** | arXiv preprint (submitted May 2026) |
| **arXiv** | https://arxiv.org/abs/2605.17366 |
| **Submitted** | 2026-05 |
| **Code** | `code/TGQ-Former/` |

---

## 方法概述 / Method Summary

电商商品图片常包含促销贴片（如"爆款推荐"角标）、杂乱背景及无关视觉元素，导致视觉 embedding 中混入大量噪声特征，严重降低 I2I（Item-to-Item）检索的精度。主流 MLLM 式管道使用冻结视觉编码器 + 轻量连接器（Q-Former）将视觉 token 汇聚后送入 LLM，但连接器必须在噪声环境下"选择性聚合"视觉 token，这对现有 Q-Former 是巨大挑战。

本文提出 **TGQ-Former（Text-Guided Q-Former）**，利用商品结构化元数据（标题、品类、属性）作为语义锚点引导视觉 token 提取：

1. **Hybrid-Query Connector（混合查询连接器）**：将查询分为两路——
   - *Metadata-Anchored Query（元数据锚定查询）*：以结构化商品文本编码为 key/value，Q-Former 查询通过交叉注意力聚焦于文本描述对应的视觉区域（如商品主体），过滤促销贴片和背景。
   - *Exploratory Query（探索性查询）*：保留自由视觉查询，捕捉文本未描述的补充视觉线索（如材质纹理、颜色细节）。
2. **Reliability-Aware Dual-Gated Vector Modulation（可靠性感知双门控向量调制）**：动态评估两路查询输出的可靠性，通过可学习门控权重自适应融合两路特征，在噪声图像输入下降低元数据锚定查询的权重，转而依赖探索性视觉特征。
3. 最终 visual embedding 与文本 embedding 对齐后，用于大规模全量 I2I 检索。

**故事弧线：** 电商商品图片充斥噪声视觉元素，现有 Q-Former 无法区分有效视觉特征与噪声 → TGQ-Former 以结构化文本为向导引入"锚定+探索"双路查询，并通过可靠性门控自适应融合，在大规模真实电商数据集上 H@100 平均提升 6.04%。

---

## 创新性分析 / Innovation Analysis

- **新颖点1 — 文本引导视觉抽取**：以商品元数据（结构化文本）作为 Q-Former 的语义锚点，精确定位有效视觉区域，这在 MLLM 连接器设计中属首次针对电商噪声图像的专项设计。
- **新颖点2 — 双路查询架构**：同时保留"有指导的精确抽取"和"无指导的全面探索"，两路互补，避免纯文本引导时遗漏关键视觉细节。
- **新颖点3 — 可靠性门控自适应融合**：不固定两路权重，而是根据当前样本图像质量动态调整，对图像噪声具有鲁棒性。
- **与前作区别**：标准 Q-Former（BLIP-2）采用统一查询，无元数据引导；MLP 连接器直接平均视觉 token，无法滤噪。TGQ-Former 专为"文本丰富但图像噪声"的电商场景定制，并针对 I2I 检索而非生成优化。

---

## 关键指标 / Key Metrics

| Dataset | Metric | This Work vs. Baseline |
|---------|--------|----------------------|
| Large-scale e-commerce (JD.COM internal, full-pool) | Hit Rate@100 (H@100) | **+6.04%** avg vs. strong connector baselines |
| — | vs. end-to-end MLLMs | Consistently outperforms |

---

## 评分 / Score

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 24 | 30 | 文本引导双路查询 + 可靠性门控，在 Q-Former 适配电商场景上有明确创新 |
| Experimental SOTA Delta | 12 | 15 | 大规模真实电商数据集 H@100 +6.04%，vs. 强基线和端到端 MLLM |
| Experimental Quality / Ablations | 12 | 15 | 大规模真实数据集 + 全量检索评测，Tsinghua+JD.COM 工业合作保证数据质量 |
| Efficiency | 7 | 10 | 冻结视觉编码器 + 轻量连接器，不全量微调 MLLM |
| Generalization | 4 | 5 | 多类目电商数据验证，但数据集内部 |
| Domain Relevance (ecom + governance) | 22 | 25 | 多模态电商 I2I 检索，与商品推荐/类目理解直接相关 |
| **Total** | **81** | **100** | |
