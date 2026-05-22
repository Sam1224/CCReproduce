# TGQ-Former (Toy Reproduction)

本目录是对论文 **"Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation" (arXiv:2605.17366)** 的可运行复现（**toy 级别**），目标是把论文核心思想落到一套可训练/可评测的 PyTorch pipeline 上。

## 核心复现内容

- **混合查询 Q-Former (HybridQueryQFormer)**：将视觉 token 提取分为两路：
  - *MetadataStream*：用结构化文本（类目/品牌/属性）初始化 Q 向量，通过交叉注意力聚焦商品相关图像区域。
  - *ExploratoryStream*：用可学习 token 初始化，自由探索细粒度视觉特征。
- **RDGVM（可靠性感知双门控向量调制）**：根据图像 patch 的统计特征（方差/熵）估计视觉可靠性，自适应调控两路流的融合权重——噪声大时提升元数据引导权重。
- **对比学习训练**：In-batch InfoNCE + Triplet Margin Loss。
- **评测指标**：H@10 / H@100（全量检索池 Hit Rate）。

> **原论文与 toy 复现的主要差异**：
> - 原论文使用大规模真实工业数据；本复现使用合成 toy 数据（200 产品 × 4 图像）。
> - 原论文视觉编码器为冻结大型 ViT/CLIP；本复现用轻量 CNN（4 层）。
> - 噪声图像用随机 patch 替换模拟，非真实促销遮罩。

## 目录结构

```
TGQFormer/
├── tgqformer/
│   ├── __init__.py
│   ├── dataset.py   — 合成电商 I2I 数据集（带噪声图像）
│   ├── model.py     — TGQFormer 全模型（ImageEncoder + TextEncoder + HybridQFormer + RDGVM）
│   ├── losses.py    — 对比学习损失（InfoNCE + Triplet）
│   └── metrics.py   — H@K / Recall@K 评测
├── train.py         — 训练脚本
├── eval.py          — 评测脚本
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练（toy，约 2 分钟 CPU）
python train.py --epochs 10 --batch_size 64 --output_dir runs/tgqformer

# 评测
python eval.py --ckpt runs/tgqformer/model.pt
```

## 关键实现与论文对应

| 论文组件 | 代码实现 |
|---------|---------|
| Hybrid-Query Connector | `tgqformer/model.py::HybridQueryQFormer` |
| Metadata-Anchored Stream | `HybridQueryQFormer.meta_xattn` |
| Exploratory Stream | `HybridQueryQFormer.explore_xattn` |
| RDGVM | `tgqformer/model.py::RDGVM` |
| Reliability estimation | `RDGVM._patch_reliability` (patch variance + entropy) |
| Dual-gated modulation | `RDGVM.gate_meta / gate_explore` |
| InfoNCE loss | `tgqformer/losses.py::ContrastiveLoss` |
| H@100 metric | `tgqformer/metrics.py::hit_rate_at_k` |
