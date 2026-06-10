# SuperFashion (Toy Reproduction)

- Paper: **SuperFashion: Attribute-Specific Fashion Retrieval with Superpixel Tokens**
- arXiv: https://arxiv.org/abs/2606.10697

## What’s implemented

这是一个 **toy but runnable** 的 PyTorch 复现，聚焦论文的核心思想：

- **Superpixel tokens**：用 SLIC 超像素把图像切成更贴合物体边界的区域，每个区域作为一个视觉 token。
- **Attribute-conditioned retrieval**：把“目标属性”（如 collar/sleeve/pattern）作为条件 token，与视觉 tokens 一起输入 Transformer，输出用于检索的属性特定表征。
- **End-to-end pipeline**：合成服饰 toy 数据集（可控属性）、训练脚本、测试脚本（mAP）。

## Limitations (vs. the full paper)

- 原论文在真实服饰数据集（FashionAI、DARN 等）上评测并包含更完整的模型/训练细节；这里用合成数据验证“超像素 token 更利于属性特定检索”的机制直觉。
- toy 数据的 mAP 数值不代表原论文结果，仅用于 sanity-check 与对比 patch-token baseline。

## Quickstart

```bash
cd 2026-06-10/SuperFashion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 训练（patch tokens）
python train.py --tokenizer patch --out_dir runs/patch
python test.py  --tokenizer patch --ckpt_dir runs/patch

# 训练（superpixel tokens）
python train.py --tokenizer superpixel --out_dir runs/superpixel
python test.py  --tokenizer superpixel --ckpt_dir runs/superpixel
```

通常在该 toy 设置下，`superpixel` 会比 `patch` 的总体 mAP 更稳定。