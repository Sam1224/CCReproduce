# OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation |
| **Authors** | Jiakai Tang, Sunhao Dai, Kun Wang, Zhiluohan Guo, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Xu Chen, Jun Xu |
| **Affiliations** | Renmin University of China; Shopee Pte. Ltd.; Nanyang Technological University |
| **arXiv** | — (industry paper, presented at RecSys/KDD 2026) |
| **Submitted** | 2026-06-17 window |
| **Domain Tags** | e-commerce, ranking, multi-task learning, transformer, Shopee |
| **Code** | `code/OneRank/` (existing reproduction) |

---

## 方法概述 / Method Summary

现有电商排序系统多采用"独立任务塔 + 后融合"的 DCN/DIN 范式，各任务间梯度耦合严重，难以适配规模化 Transformer 架构。OneRank 提出 Transformer-native 多任务统一排序框架：将所有任务特征、交互和打分统一在单一 Transformer 主干中处理，通过**任务私有通道**（Task-Private Channels）和**梯度脱钩机制**（Gradient Decoupling）避免任务间干扰，同时保持参数共享的泛化优势。最终在 Shopee 主排序上完成大规模线上 A/B 实验。

**Story arc**: 多任务排序中任务冲突与梯度干扰是业界痛点，Transformer 架构带来容量优势但多任务适配困难 → 设计 Transformer-native 任务私有通道 + 梯度脱钩，实现统一高效多任务排序。

**Key components**:
1. **Unified Transformer Backbone**: 所有特征（用户、物品、上下文）统一经 Transformer 编码
2. **Task-Private Channels**: 每个任务在 Transformer 内拥有私有 Query/Key/Value 子空间
3. **Gradient Decoupling**: 反向传播时任务梯度相互隔离，防止负迁移
4. **Match-Style Scoring**: 使用匹配式打分（非点积拼接），提升任务头的泛化性

---

## 创新性分析 / Innovation Analysis

**vs. prior work**:
- 相比 MMOE、PLE 等门控多任务方法，OneRank 是首个将 Transformer 自注意力深度集成到多任务排序的工业系统
- 任务私有通道 + 梯度脱钩组合首次在 Shopee 级别平台验证
- 线上 A/B 实验证明 Transformer-native 多任务范式优于 DNN-based 基线

**Novelty assessment**: 工程创新扎实，范式有引领作用；Shopee 大流量平台背书，可信度高。

---

## 关键指标 / Key Metrics

| Dataset/System | Metric | OneRank | Baseline |
|---------------|--------|---------|----------|
| Shopee Main Ranking (online A/B) | GMV | significant lift | — |
| Shopee Main Ranking (online A/B) | CTR/CVR | positive | — |
| Inference | Efficiency | maintained | — |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 27 | 30 | Transformer-native 多任务排序，工业创新扎实 |
| Experimental SOTA delta | 12 | 15 | 线上 A/B 有收益，但数值未完全公开 |
| Experimental quality / ablations | 13 | 15 | 真实线上实验 + 消融 |
| Efficiency | 9 | 10 | 线上延迟可接受 |
| Generalization | 3 | 5 | 仅单一 Shopee 数据集 |
| Domain relevance | 16 | 25 | 电商主排序，间接相关内容治理 |
| **Total** | **80** | **100** | 强电商工业论文，达到复现阈值 |
