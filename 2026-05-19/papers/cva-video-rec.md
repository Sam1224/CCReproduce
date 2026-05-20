# CVA: Compressed Video Aggregator for Efficient Micro-Video Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Compressed Video Aggregator: Content-driven Module for Efficient Micro-Video Recommendation |
| **Authors** | Yang Xiao, Huiyuan Chen, Kaiyuan Deng, Chao Jiang, Zinan Ling, Ruimeng Ye, Xiaolong Ma, Bo Hui |
| **arXiv** | https://arxiv.org/abs/2605.08810 |
| **Submitted** | May 9, 2026 |
| **Domain** | Micro-Video Recommendation · Content-Driven Embedding · VFM · Efficiency |

---

## 方法概述 / Method Summary

大规模视频推荐系统（如 TikTok、抖音）需要高效地将视频内容信息融入推荐模型。现有方法直接将 Video Foundation Model（VFM）如 CLIP/InternVideo 与推荐模型联合训练，计算成本极高，且视频帧冗余严重。

**CVA（Compressed Video Aggregator）** 提出一种**解耦+压缩**策略：
1. **解耦：** CVA 将视频内容理解从用户偏好学习中解耦 — VFM 参数冻结，只训练轻量聚合器
2. **帧重采样：** 发现原始基准中帧采样过粗，利用 CLIP 和视频标题做语义重采样（选更具代表性的关键帧）
3. **Latent Reasoning 聚合：** CVA 通过无需 cross-attention projection 的隐式推理，将多帧 VFM embeddings 压缩成紧凑的视频表示
4. **即插即用：** CVA 作为轻量模块可接入任意推荐 backbone

---

## 故事弧线 / Story Arc

> 微视频推荐中的 VFM 集成因计算代价高且帧冗余问题而难以规模化部署 → CVA 通过冻结 VFM + 轻量压缩聚合器 + CLIP 引导的语义关键帧重采样 → 训练时间和 GPU 内存降低数量级，同时保持并提升推荐效果

---

## 创新分析 / Innovation Analysis

- 系统性识别了"帧冗余"问题，并提出基于 CLIP+标题的语义关键帧选择，这一贡献本身就有实用价值
- 解耦策略（冻结 VFM）大幅降低了训练成本，使大规模视频推荐落地成为可能
- 模块化设计（plug-and-play）意味着可直接用于 DIN、SASRec 等现有推荐框架

---

## 关键指标 / Key Metrics

| Dataset | Metric | CVA vs Baselines |
|---------|--------|-----------------|
| MicroLens | Recall, NDCG | Consistent gains |
| Short-Video | Recall, NDCG | Consistent gains |
| Training Time | GPU hours | 数量级降低 (orders of magnitude) |
| GPU Memory | Peak memory | 数量级降低 |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | 语义关键帧重采样 + 解耦聚合新颖，但整体思路较直观 |
| Experimental SOTA delta | 12 | 15 | 两个基准一致提升，效率优势显著 |
| Experimental quality | 11 | 15 | 多 backbone 比较，但缺少更多 ablation |
| Efficiency | 10 | 10 | 数量级效率提升，满分 |
| Generalization | 4 | 5 | 两个数据集，多 backbone |
| Domain relevance | 22 | 25 | 微视频推荐核心问题 |
| **Total** | **79** | **100** | |

