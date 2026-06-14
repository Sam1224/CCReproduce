# Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution |
| **ArXiv ID** | [2606.13668](https://arxiv.org/abs/2606.13668) |
| **Authors** | Dimitri Kachler, Damien Sileo, Pascal Denis |
| **Affiliation** | Inria / Université de Lille |
| **Submitted** | 2026-06-11 |
| **Source** | HuggingFace June 13 daily listing |
| **Bucket** | STRONG — 数据归因、数据过滤、蒸馏、数据治理 |
| **Code** | `2026-06-13/Influcoder/` |

---

## 方法概述 / Method Overview

**故事弧线：** 基于梯度的 influence function 能精确定位导致模型有害行为的训练样本，但在大规模数据上计算代价极高（数小时+数百 GB）。→ Influcoder 将"influence 排序"作为蒸馏目标，训练轻量 encoder 将样本映射为 influence embedding，之后用 embedding 余弦相似度替代梯度计算进行快速归因检索（32 秒 vs 2285 秒）。

**三步流程：**
1. **Ground-Truth Influence 生成**：对目标 LLM（decoder），用 LoRA 子空间梯度 + CountSketch 压缩，计算 query 对 pool 样本的 influence ranking；
2. **Influence Embedding 蒸馏**：训练 encoder，使其产出的 influence embedding 的余弦相似度拟合上述 ground-truth influence ranking（listwise ranking loss）；
3. **在线归因推理**：新 query → encoder 生成 embedding → ANN 检索最相关训练样本，用于归因/过滤/数据质量审核。

**核心创新：**
- 以"influence 排序"（而非 loss/输出）作为蒸馏监督信号，保留了 influence 方法的因果语义；
- LoRA 子空间 + CountSketch 双重压缩，让 ground-truth 生成在单卡可行；
- 推理期完全无梯度，embedding 可预计算并持久化。

---

## 关键指标 / Key Metrics

| 方法 | 毒性过滤 AUPRC (Het) | 推理时间 | 存储 |
|------|---------------------|---------|------|
| LESS（梯度法基线）| 0.7075 | 2248.5s | 320MB |
| LoGRA | — | 3060s | 2.24GB |
| **Influcoder (dim=768)** | **0.6981** | **32.1s** | **30MB** |

速度提升：**70× faster**；存储降低：**10×**；AUPRC 损失：~1.4%（可接受）

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 24 | 30 | influence ranking 作为蒸馏目标是创新点；LoRA+CountSketch 组合实用 |
| 实验指标 | 12 | 15 | 毒性过滤 AUPRC 接近梯度法，速度质量 tradeoff 清晰 |
| 实验质量 | 12 | 15 | 多 ablation（dim/batch/encoder arch），与多基线对比 |
| 方法效率 | 10 | 10 | 70× 提速，10× 存储降低，核心贡献 |
| 方法泛化性 | 3 | 5 | 主要验证 toxicity；其他有害内容场景需额外验证 |
| 领域相关性 | 21 | 25 | 大规模数据归因/清洗/有害样本过滤，对内容治理数据管道极有价值 |
| **Total** | **82** | **100** | |

**复现路径：** `2026-06-13/Influcoder/`

---

## Story Arc

> **现状不足：** 精确数据归因（influence function）太慢太贵，无法在工业级数据上实时运行；粗略过滤方法缺乏因果依据。  
> **解法：** 一次性离线算 influence ranking → 蒸馏到轻量 encoder → 在线用 embedding 相似度代替梯度 → 70× 加速，归因精度基本保留。
