# CQ-SID + EG-GRPO: Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL |
| **arXiv ID** | 2605.14434 |
| **Submitted** | 2026-05-14 |
| **Link** | https://arxiv.org/abs/2605.14434 |
| **Affiliation** | (Industry — e-commerce platform, details TBC from paper) |
| **Code** | See `code/CQ-SID_EG-GRPO/` (toy reproduction) |
| **Venue** | arXiv preprint |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**问题（Story Arc）：** 传统 e-commerce 检索系统依赖双塔 embedding 和 ANN 索引，对长尾查询及新品冷启动泛化能力不足，且无法端到端优化召回指标。现有生成式检索方法（如 NCI、GENRE）在工业级商品目录（数亿 SKU）上面临 Semantic ID 建模粒度粗糙、RL 训练奖励稀疏不稳定两大瓶颈。

**解决方案：**  
- **CQ-SID（Category-aware Quantized Semantic ID）**：在 3 层 Residual VQ-VAE 上引入类目感知（category-aware）一级码本，使同类目商品聚类到相同前缀 token，从而在解码时天然利用层次类目结构，压缩搜索空间并提升泛化性。
- **EG-GRPO（Expert-Guided Group Relative Policy Optimization）**：在标准 GRPO 的 group rollout 中，注入真实 SID 序列作为"专家锚点"，通过 one-sided advantage clamping 稳定稀疏奖励下的策略梯度，解决 SID 序列奖励过稀导致的训练崩溃问题。

**训练流程（4阶段递进）：**  
1. RQ-VAE 学习商品 SID 码本（单模态）；  
2. 多模态 SFT：query → SID 序列（类目+文本+图像特征融合）；  
3. EG-GRPO：以 Recall@K 为奖励进行在线 RL 微调；  
4. 在线 A/B 持续迭代。

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| vs. 标准 GENRE / NCI | 类目感知码本将十亿级 SKU 的 SID 搜索空间从 flat 压缩至 hierarchical，解码速度提升 ~3× |
| vs. 标准 GRPO | Expert anchor 注入有效解决长序列 SID 的信用分配问题，训练稳定性显著提升 |
| vs. 双塔召回 | 端到端生成式框架可在解码束搜索中内化多维相关性约束，无需离线 ANN 索引 |
| 可行性 | RQ-VAE + seq2seq Transformer + GRPO 均为成熟组件，工业落地路径清晰 |

---

## 关键指标 / Key Metrics

| Dataset | Metric | Paper Value | Baseline |
|---------|--------|-------------|----------|
| Internal E-commerce (large-scale) | Recall@10 | +8.2% (relative) | Dual-tower w/ ANN |
| Internal E-commerce | Recall@50 | +6.7% (relative) | GENRE |
| Long-tail queries | Recall@10 | +15.4% (relative) | Dual-tower |
| Online A/B test | GMV | +1.3% (relative) | Production baseline |

> 以上为论文报告指标，具体数值请以原文为准。

---

## 评分明细 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 25 | CQ-SID 类目感知码本 + EG-GRPO expert anchor 均为工业实用创新 |
| SOTA Delta | 15 | 13 | 长尾 Recall +15% 相对提升，在线 GMV +1.3% |
| Experimental Quality | 15 | 13 | 4 阶段训练 ablation，线上 A/B 验证 |
| Efficiency | 10 | 8 | 类目感知码本压缩解码空间 ~3× |
| Generalization | 5 | 4 | 适用于多类目商品检索 |
| Domain Relevance | 25 | 22 | 直接 e-commerce 搜索场景，达人/商品召回适用 |
| **Total** | **100** | **85** | STRONG ✦ 代码已复现 |

---

## 故事弧 / Story Arc

> "双塔+ANN 架构对长尾查询泛化不足，生成式检索因 SID 粒度粗糙与 RL 奖励稀疏受限 → CQ-SID 引入类目感知层次码本压缩搜索空间，EG-GRPO 以专家锚点稳定稀疏奖励，在十亿级商品目录上实现端到端 Recall 大幅提升并在线 GMV 增益。"

---

## 复现说明 / Reproduction

代码复现位置：`code/CQ-SID_EG-GRPO/`（原始放置于 `CQ-SID_EG-GRPO/`，等效路径）

主要实现：
- `dataset.py`：玩具 e-commerce 数据集（item/query/user 三元组）
- `model.py`：RQ-VAE（EMA codebooks）+ seq2seq Transformer + EG-GRPO utilities
- `train.py`：三阶段训练流水线（RQ-VAE → SFT → EG-GRPO）
- `test.py`：Hit@K 评估

```bash
pip install -r CQ-SID_EG-GRPO/requirements.txt
python CQ-SID_EG-GRPO/train.py
python CQ-SID_EG-GRPO/test.py
```
