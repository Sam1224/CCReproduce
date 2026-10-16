# SAM-D2Q (Toy Reproduction)

- Paper: **SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce**
- arXiv: https://arxiv.org/abs/2609.04961

## What is implemented

这是一个 **toy but runnable** 的 PyTorch 复现，目标是尽量保留论文最关键的三段式思路，而不是只写一个名字相似的 baseline：

1. **Stage 1: Multimodal SFT**
   - 输入商品标题与 toy 商品图像。
   - 用轻量文本编码器 + 图像编码器预测文档扩展词（expanded query terms）。
   - 对应论文中“用多模态 SFT 先把商品标题 / 图片 / 用户查询对齐起来”。

2. **Stage 2: Counterfactual Visual Augmentation**
   - 人为把标题里的视觉属性词（颜色 / 风格 / 材质）mask 掉。
   - 模型仍要从图像中恢复这些属性并补全扩展词。
   - 对应论文里用 counterfactual 方式逼模型真的“看图补词”，而不是抄标题。

3. **Stage 3: Business-aware Preference Optimization**
   - 对同一商品采样多组候选扩展词。
   - 用一个简化的 group-relative reward 做 policy gradient。
   - reward 同时考虑：语义一致性、商业价值、视觉属性恢复。
   - 对应论文中用 GRPO 将生成结果对齐到搜索业务目标。

4. **完整 pipeline**
   - toy 电商商品数据自动构造
   - toy 商品图像绘制
   - 训练脚本（三级训练）
   - 测试脚本（title-only vs. stage1 vs. full）
   - 输出扩展词与检索指标

## Files

- `data.py`: toy 商品、图像、训练样本、counterfactual 样本、query 集合
- `model.py`: 多模态扩展词生成模型与简化 GRPO 采样/奖励逻辑
- `train.py`: 三阶段训练脚本
- `test.py`: 检索评测脚本
- `requirements.txt`: 依赖

## Alignment to the paper

本复现尽量对齐原文的“工业逻辑”：

- Boolean retrieval / lexical expansion：这里把文档扩展建模成 **term-level expansion**，而不是 dense retrieval。
- Multimodal grounding：图像里显式携带标题省略的视觉属性。
- Counterfactual augmentation：训练时会 mask 掉标题里的视觉词。
- Business-aware optimization：reward 会偏好更高 demand / conversion proxy 的 query 词组合。

## Limitations

- 原文是工业级 MLLM + 离线索引增强；这里是轻量 toy 模型，便于本地快速跑通。
- 原文 Stage 3 使用 GRPO；这里实现的是 **简化版 group-relative policy gradient**，保留“组内相对奖励”思想，但不是全文 1:1 训练框架。
- 原文在线指标是 GMV / Pay Count / CTR / CVR；这里用 toy retrieval 指标模拟：`Recall@1`、`Relevant@1`、`TotalQuality@1`、`ZeroResultRate`。
- 原文使用真实商品图和业务日志；这里的数据、图像、query demand 均为合成。

## Quickstart

```bash
cd 2026-10-16/SAM-D2Q
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 训练
python train.py --out_dir runs/sam_d2q

# 评测
python test.py --ckpt runs/sam_d2q/model.pt
```

## Expected behavior

- `title-only` baseline 会因为标题缺少视觉属性而丢失召回。
- `stage1` 通常比 `title-only` 更强。
- `full`（stage1 + stage2 + stage3）通常会进一步提升 `Relevant@1` 与 `TotalQuality@1`，并降低 `ZeroResultRate`。

这是一个用于 **验证论文核心机制** 的最小闭环复现，而不是工业等价实现。
