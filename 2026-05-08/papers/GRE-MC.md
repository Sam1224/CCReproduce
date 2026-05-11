# GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

- **arXiv**: https://arxiv.org/abs/2605.00670
- **作者**: Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He
- **领域标签**: `Multimodal Recommendation` `Graph` `Modality Completion`

## 方法概述
现实多模态推荐数据常缺模态 (传感器故障 / 标注缺失 / 隐私)。GRE-MC 用 **modality-aware subgraph retrieval** 在物品图上找语义相近的子图作为缺失模态的"补全证据"，再用一个带全局注意力的 Graph Transformer 对图做联合编码；引入 **learnable sparse-routing codebook** 作正则。

## 故事
多模态推荐 → 真实场景模态不全 → 用图结构上的"邻居证据"补出缺失模态特征。

## 创新性分析
- 模态补全 + subgraph retrieval 的耦合是新组合；
- sparse routing codebook 类似 VQ 思想，能稳定训练；
- 缺失模态作为问题入手，比 naive multimodal recsys 更贴合工业场景。

## 关键指标
- 摘要未给具体数据集和指标 — 评分中实验项保守给。

## 评分
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 20 / 30 | subgraph retrieval + modality completion 组合 |
| 实验指标 | 9 / 15 | 摘要无数字 |
| 实验质量 | 9 / 15 | 待论文细节 |
| 效率 | 6 / 10 | 图 transformer 中等代价 |
| 泛化性 | 3 / 5 | 多模态实验泛化 |
| 相关性 | 22 / 25 | 多模态推荐 = 电商主战场 |
| **合计** | **69** |
