# LazFormer — Scaling Transformers for Industrial Recommendation via Transferable Generative Pre-training（toy reproduction）

- Paper: **LazFormer: Scaling Transformers for Industrial Recommendation via Transferable Generative Pre-training**
- 目标：在一个**toy 但可运行**的合成数据集上，复现论文的核心结构：
  1) **Generative pre-training**（请求条件的自回归序列建模）
  2) **Transferable residual adapter**（冻结 Transformer blocks 的 attention/FFN，仅训练 adapter + embeddings + head 做迁移）
  3) **Request-aware ranking**（简化版 *coarse-to-fine compression* + *sparse attention mask*）
  4) **Ranking fine-tuning**（候选集 softmax 排序训练）

> 说明：工业论文中的真实日志与超大模型不可公开复现；本目录关注“结构与训练范式”的可运行最小实现。

## 文件说明

- `data.py`：合成数据
  - **Source domain**：用于 generative pre-training 的请求条件序列（user history → next item）
  - **Target domain**：用于 ranking fine-tuning 的 (request, history, candidates) 排序样本（包含 domain shift）
- `model.py`：
  - `LazFormer`：小型 Transformer + **Residual Adapter** + **稀疏注意力 mask**
  - Baseline（`train.py`/`test.py`使用）：**last-M 截断**的 dense Transformer ranker（无预训练/无 adapter/无稀疏 mask）
- `train.py`：两阶段训练
  - `pretrain`：source domain 自回归 next-item 预测
  - `rank`：target domain 排序微调（冻结 attention/FFN，训练 adapter + embeddings + rank head）
- `test.py`：在 target domain 测试集上对比 **Baseline vs LazFormer**（Recall/NDCG）

## 快速运行

在该目录下执行：

```bash
python train.py
python test.py
```

（CPU 下几十秒内可跑完；数值会随随机种子略有波动。）

## 关键实现点（论文思想 → 本 toy 代码）

1) **Generative pre-training**：
   - 输入 token：`[REQ] + item_1 + ... + item_N`
   - 训练目标：位置 `t` 预测 `item_{t+1}`（`[REQ]` 位置预测 `item_1`）
   - 代码：`LazFormer.pretrain_loss(...)`

2) **Transferable residual adapter**：
   - 每层 FFN 后插入 bottleneck adapter：`x <- x + s * Up(Act(Down(LN(x))))`
   - ranking 阶段：冻结 Transformer backbone，仅训练 adapter（+ 轻量 head）
   - 代码：`ResidualAdapter`、`LazFormer.freeze_backbone_for_rank()`

3) **Request-aware ranking（简化版 coarse-to-fine + sparse attention）**：
   - Coarse：用 `request` 向量与历史 item embedding 的相似度取 Top-M 历史作为“压缩后的关键 history”
   - Fine：构造 batch-wise **稀疏 self-attention mask**，限制 candidate token 只关注 `{REQ + Top-M history + candidates}`
   - 代码：`LazFormer._build_sparse_attn_mask(...)`

4) **Ranking fine-tuning**：
   - 候选集 softmax：`CE( scores(request, history, candidates), label_pos_index )`
   - 代码：`train.py::train_rank(...)`

## 局限

- toy 数据分布是可控合成的；指标仅用于验证训练/结构“可学习且能提升”，不能与论文工业结果对标。
- 稀疏注意力实现为了清晰可读，使用了 PyTorch `MultiheadAttention` 的 3D mask（batch*head 维度展开），并未做工程级优化。
