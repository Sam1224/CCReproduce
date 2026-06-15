# PAD (Toy Reproduction)

- Paper: **When Recommendation Denoising Meets Popularity Bias: Understanding and Mitigating Their Interaction**
- arXiv: https://arxiv.org/abs/2606.14046

## What’s implemented

这是一个 **toy but runnable** 的 PyTorch 复现，聚焦论文的核心公式与可验证现象：

- Base denoising（RCE 风格）：对每个样本构造 loss-based 权重，正样本权重 `w = \hat{y}^α`（负样本对称为 `(1-\hat{y})^α`），对应论文 Eq.(14)。
- PAD gate：用 item popularity 构造 gate 系数 `s_i=(pop(i)/max_pop)^η`，并将权重修正为 `w_PAD = (1-s_i) + s_i*w`，对应论文 Eq.(13)(15)。
- Toy 数据：合成 implicit feedback + head-item noise（更贴近“曝光导致 head 噪声、tail 更稀疏更难拟合”的设定）。
- 评测：Recall/NDCG（单一 held-out 正样本）+ Coverage@K 与 Gini-Div（越大越多样），用于观察“去噪 ↔ 多样性”与 PAD 的 trade-off。

## Limitations (vs. the full paper)

- 原文在 MovieLens/Amazon-Book/Yelp + GMF/NeuMF/LightGCN 上做系统实验并比较多个 denoiser（RCE/TCE/DeCA/…）；这里用 **单一 MF backbone** + **RCE-style base denoiser** 做最小闭环复现。
- toy 数据无法复现真实数据分布与所有 ablation，但代码接口与公式对齐，可用于后续接入真实数据集。

## Quickstart

```bash
cd 2026-06-15/PAD
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train & evaluate ERM / BaseDenoise / PAD
python train.py --out_dir runs/pad

# Re-evaluate from checkpoint
python test.py --ckpt_path runs/pad/ckpt.pt
```
