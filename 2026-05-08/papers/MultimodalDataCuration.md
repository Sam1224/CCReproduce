# Multimodal Data Curation Through Ranked Retrieval (SNS + EEE)

- **arXiv**: https://arxiv.org/abs/2605.01163
- **作者**: Pratyush Muthukumar, Harshil Kotamreddy, Sarah Amiraslani, Tomo Kanazawa, Ramani Akkati, Shaan Jain, Andrew Mathau
- **领域标签**: `Multimodal Embeddings` `Data Curation` `数据清洗` `Modality Gap`

## 方法概述
针对多模态共享嵌入空间两个常见问题——(1) embedding 偏 modality 而不是语义；(2) 监督对齐对噪声敏感——提出 **Symmetric Nucleus Subsampling (SNS)** 在 pair 级别裁剪到互相支撑的子片段；**Expert Embedding Engine (EEE)** 用一个学得的投影网络融合多个 embedding expert，并配 bias-aware 损失抑制 modality-driven 分离。

## 故事
跨模态检索 + 多源数据 blend → modality gap + 监督噪声相互放大 → SNS 修剪训练对，EEE 融合 expert + 偏置感知约束。

## 创新性分析
- 同时处理"训练对"和"嵌入模型"两层是合理的二阶设计；
- "Symmetric nucleus" 命名借自 nucleus sampling，把它扩展到 pair-level 是清晰的类比；
- 数据 curator 视角让方法可作为大规模 pretraining 的预处理插件。

## 关键指标
- Modality gap 平均压缩 **>90%** 相对 base expert；
- 数据 blend 比 stratified sampling / 传统 curation 在下游模型性能上更优。

## 评分
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 22 / 30 | SNS + 偏置感知投影是新颖组合 |
| 实验指标 | 12 / 15 | "modality gap 收窄 >90%" 是清晰的可量化收益 |
| 实验质量 | 11 / 15 | 多 datablend 对比 |
| 效率 | 6 / 10 | EEE 需要训练投影网，但推理便宜 |
| 泛化性 | 4 / 5 | 设计就是要跨数据集泛化 |
| 相关性 | 22 / 25 | 电商大规模数据清洗痛点直接命中 |
| **合计** | **77** |
