# TIGER-FG (Toy Reproduction)

本目录是对论文 **"TIGER-FG: Text-Guided Implicit Fine-Grained Grounding for E-commerce Retrieval" (arXiv:2605.18434)** 的可运行复现（**toy 级别**），目标是把论文核心思想落到一套可训练/可评测的 PyTorch pipeline 上：

- **IMMR 任务定义**：query 是 **crop 图**（局部目标），candidate 是 **full item image + structured text**。
- **Text-guided implicit grounding**：在 item encoder 中，用文本作为语义引导对全图 patch 进行注意力聚合，得到 **target-focused item embedding**，避免显式检测/框选的线上开销。
- **Dual distillation（简化实现）**：
  - *Spatial-relational distillation*：用 toy 数据自带的 box mask 作为 teacher 的“区域先验”，蒸馏学生的 attention / patch-relation。
  - *Similarity-distribution distillation*：蒸馏 batch 内 query–item 相似度分布（soft targets）。

> 说明：原论文使用了大规模真实电商数据、结构化文本、以及更复杂的视觉/文本 backbone。本复现为了可在本环境直接跑通，采用了 **合成 toy 数据** 与 **轻量模型**，但接口设计尽量贴近真实 IMMR 训练流程（query-side / item-side encoder、蒸馏 loss、Recall@K 评测）。

## 目录结构

- `tiger_fg/`
  - `dataset.py`：合成 IMMR toy 数据集（full image + box + text；crop query）。
  - `model.py`：简化 TIGER-FG（image encoder + text encoder + text-guided cross-attn 聚合）。
  - `losses.py`：对比学习 + 双蒸馏损失。
  - `metrics.py`：Recall@K。
- `train.py`：训练脚本。
- `eval.py`：评测脚本。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 训练（toy）
python train.py --output_dir runs/toy --epochs 3 --batch_size 128

# 评测
python eval.py --ckpt runs/toy/model.pt
```

## 关键实现与论文对应

- **Text-guided implicit grounding**：`tiger_fg/model.py::ItemEncoder` 中用文本 token 作为 query，对图像 patch 进行 cross-attention 聚合，输出 item embedding。
- **Spatial-relational distillation**：`tiger_fg/losses.py::spatial_relational_distillation`。
- **Similarity-distribution distillation**：`tiger_fg/losses.py::similarity_distribution_distillation`。

## 已知差异 / TODO（与原论文一致性声明）

- backbone 未使用论文中的大规模预训练模型；toy 任务难度低于真实电商。
- distillation 实现为“可运行的简化版”，未复现论文全部细节（例如更精细的空间关系定义、更多训练 trick）。如需进一步对齐：
  - 可以把 toy 数据替换为真实电商图搜数据（包含 query crop、candidate full image、structured text、训练box）。
  - 替换 image encoder 为 ViT/CLIP，text encoder 为更强的文本编码器。

