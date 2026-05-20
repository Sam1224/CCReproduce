# SAGE（Toy Reproduction）

本目录是对论文 **"SAGE: Scalable Automatic Gating Ensemble for Confident Negative Harvesting in Fraud Detection" (arXiv:2605.20157)** 的 *toy 级别* 可运行复现，目标是把论文核心流程落到一套可训练/可评测的 pipeline 上，便于在电商内容生态 & 达人治理场景下复用其思想：

- **PU Learning**：只有少量正样本（异常/欺诈/违规）有标签，大量样本处于 Unlabeled。
- **核心问题（counterfactual problem）**：如果训练负样本（正常）缺少“看起来像异常的正常边界样本”，线上容易把“边界正常人群”误判为异常（高误杀）。
- **SAGE 的核心思路**：先把 Unlabeled 里的“置信负样本”挖出来，且要保证覆盖长尾人群（通过 SimHash 分桶 + floor 采样），再用 **统计门控（Mahalanobis）+ 局部密度门控（kNN）** 的投票机制筛选。

> 说明：原论文在超大规模真实流式行为数据上验证，并报告了相对 Isolation Forest baseline 的巨大提升（+81.9pp precision、+87.2pp recall）。本复现使用合成 toy 数据，只追求**流程对齐**与**接口可复用**，不追求复现论文的绝对指标。

## 目录结构

- `sage/`
  - `dataset.py`：构造 toy PU 数据（包含一个与正类分布重叠的 edge cohort，用于模拟“超级粉丝/睡眠播放”等边界正常行为）。
  - `simhash.py`：SimHash 分桶。
  - `harvest.py`：floor-constrained 分桶采样 + 门控投票式置信负样本挖掘。
  - `gates.py`：Mahalanobis gate / kNN density gate。
  - `model.py`：简单 MLP 二分类器。
  - `metrics.py`：Precision / Recall / F1 + edge cohort 的 FPR（用于观测误杀）。
- `train.py`：训练脚本（同时跑 SAGE harvesting 与 random negatives baseline 做对比）。
- `eval.py`：评测脚本。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 训练 + 生成 report.json（会同时输出 SAGE 与 Random baseline 指标）
python train.py --output_dir runs/toy --epochs 5

# 评测（例如评测 SAGE 模型）
python eval.py --ckpt runs/toy/model_sage.pt
```

## 与原论文的一致性声明（简化点）

- **一致**：SimHash 分桶 + floor 采样；Mahalanobis + kNN 两类 gate 投票；在 PU 学习前做 negative harvesting。
- **简化**：
  - 原论文强调工程可扩展性与线上部署，本复现用 brute-force `torch.cdist` 计算 kNN 密度（仅适用于 toy）。
  - 原论文还有更复杂的“演进式”方法对比与人审流程，本复现只保留 1 个 baseline（random negatives）。

