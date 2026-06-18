# One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets |
| **Authors** | Woosung Kang, Jiwon Jeong, Jonghyeok Shin, Jeongwhan Choi, Noseong Park |
| **Affiliations** | KAIST |
| **arXiv** | — |
| **Submitted** | 2026-06-17 window |
| **Domain Tags** | sequential recommendation, synthetic data, pretraining, zero-shot, prior-fitting |

---

## 方法概述 / Method Summary

传统序列推荐模型每个数据集需独立微调，成本高且难以零样本迁移至新类目。本文将 Prior-Fitted Networks（PFN）范式首次引入序列推荐：在大量合成先验（模拟 e-commerce 用户行为的合成数据）上预训练一个通用序列推荐模型，推理时单次前向传播即可"适配"（in-context adaptation）任意新数据集，无需梯度更新（update-free）。

**Story arc**: 序列推荐迁移代价极高（每域微调 1-15 小时），新类目冷启动困难 → 合成先验预训练 + 后验预测范式，单次前向适配新域，效率提升千倍。

**Key metrics**:
- 1 分钟适配 vs 15 小时微调基线
- 未见类目零样本泛化显著

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 27 | 30 | PFN + 序列推荐的首次结合，范式创新 |
| Experimental SOTA delta | 12 | 15 | 多数据集零样本竞争力强 |
| Experimental quality / ablations | 13 | 15 | 多数据集、多基线对比 |
| Efficiency | 10 | 10 | update-free，1分钟 vs 15小时 |
| Generalization | 5 | 5 | 零样本新类目泛化 |
| Domain relevance | 9 | 25 | 序列推荐为主，与内容治理弱相关 |
| **Total** | **76** | **100** | 效率创新突出，但与创作者治理直接相关性弱 |
