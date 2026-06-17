# OneRank (Toy Reproduction)

- **Paper**: *OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation*
- **Authors**: Jiakai Tang, Sunhao Dai, Kun Wang, Zhiluohan Guo, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Xu Chen, Jun Xu (Renmin University of China · Shopee · Nanyang Technological University)
- **arXiv**: https://arxiv.org/abs/2606.16838
- **Reproduction score**: 80/100 · status: *reproduced* (no official code released)

这是一个 **toy-but-runnable** 的 PyTorch 复现，忠实还原 OneRank 的核心范式：把多任务推理**内置进 Transformer 栈内**，取消传统“编码器 F → 预测器 G”的分离。代码可在 CPU 上 < 2 分钟跑完 train + test。

## 方法概述（论文）

OneRank 是一个 Transformer-native 的多任务排序框架。前向自底向上构建**任务私有通道**：

1. **结构化 Token 化**：输入级注入相互不可见的任务 token，实现早期任务专精；
2. **任务专属编码**：用带结构化掩码的 Transformer 层堆叠；
3. **候选感知上下文化**：用情境描述符（Situational Descriptor）做候选感知聚合；
4. **多任务预测**：用可配置掩码的跨任务关系注意力做受控知识迁移，反向用**梯度脱钩**只放行对角线梯度，使跨任务注意力成为“只读记忆”，避免负迁移；
5. **匹配式打分**：用基于内积的动态打分（s = z·r）替换静态 MLP 打分头。

## 本仓库忠实复现了什么

| 论文模块 | 本复现实现 (`model.py`) |
| --- | --- |
| (1) 结构化 Token 化 | `OneRank.tokenize`：`[history | preference-anchor | 每个候选组 (candidate emb + 3 个任务 token)]`，含 item / type / task / position 嵌入 |
| (2) 任务专属编码 + 结构化注意力掩码 | `build_structured_mask`：**3 个任务 token 互不可见**、**候选组之间相互隔离**、**用户上下文因果可见**；`EncoderLayer` × L |
| (3) 候选感知上下文化 | 情境描述符（用户上下文均值）→ 每任务 query 投影 → 对该任务候选 token 做 **多头交叉注意力** → 每任务全局表征 `h_k` |
| (4) 多任务预测 + **梯度脱钩** | 对 `{h_k}` 做跨任务关系注意力，`kv = h.detach()` 使 key/value 脱钩→只有对角线/自身梯度回流；残差 + LayerNorm + FFN → `z_k` |
| (5) 匹配式打分 | `s_k^i = <z_k, r_k^i>`（逐任务、逐候选内积）+ 可学习温度 |
| 训练目标 | `train.py`：list-wise **InfoNCE** + point-wise **BCE** 的多任务联合损失 |
| 评测 | `test.py`：留出 toy split 上的逐任务 **AUC**（Mann-Whitney 估计，无 sklearn 依赖） |

## 相对完整论文的简化（toy 设定）

- **数据**：原文在 Shopee 工业日志（33M 用户 / 118M 商品 / 26.6B 曝光）上训练，候选嵌入来自上游召回的专有基础设施。这里用一个**低秩用户/物品因子的合成世界**生成可学习的 toy 数据（`data.py`），候选以 **item id** 给出、由模型在结构化 token 化时嵌入为“候选嵌入”。代码中以单行注释标出对应论文使用专有基础设施之处。
- **标签依赖**：合成的 click / cart / order 三个二元标签按相关度阈值构造，天然满足 **order ⊂ cart ⊂ click** 的嵌套依赖。
- **规模**：`d_model=64`、`L=2` 层、`N=6` 候选、~3k 训练样本，仅为验证范式与公式对齐，不复现绝对指标与全部消融。
- **未实现/留作扩展**：可配置的跨任务掩码（这里跨任务注意力为全连接 + 梯度脱钩）、线上 A/B 与大规模工程优化。

## 参考结果（CPU，默认超参，~90s）

```
epoch=20  loss=1.8999  (bce=0.4415  infonce=1.4584)   # loss 单调下降，InfoNCE 远低于 log(N)=1.79 的随机基线

=== OneRank per-task AUC (held-out toy split) ===
   click-AUC ≈ 0.70
    cart-AUC ≈ 0.72
   order-AUC ≈ 0.76
  mean  -AUC ≈ 0.73
```

> 数值会因随机性略有浮动，但三个任务 AUC 均显著高于 0.5，验证结构化 token 化 + 候选感知上下文化 + 梯度脱钩 + 匹配式打分这一前向链路确实学到了多任务排序信号。

## 如何运行

```bash
cd 2026-06-17/OneRank
pip install -r requirements.txt
python train.py        # 训练，打印逐 epoch 递减的 loss，保存 runs/onerank/ckpt.pt
python test.py         # 加载 checkpoint，打印逐任务 AUC
```

文件说明：

- `requirements.txt` — 依赖（torch、numpy）
- `data.py` — toy 数据集 / dataloader（history、preference-anchor、N 候选、3 个嵌套任务标签）
- `model.py` — OneRank 架构（结构化 token 化 / 结构化掩码编码 / 候选感知上下文化 / 梯度脱钩跨任务预测 / 匹配式打分）
- `train.py` — InfoNCE(list-wise) + BCE(point-wise) 多任务训练
- `test.py` — 逐任务 AUC 评测
