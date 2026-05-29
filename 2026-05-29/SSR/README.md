# SSR (Single-stage Sparse Retrieval) — 复现（Toy）

对应论文：**No More K-means: Single-Stage Sparse Coding for Efficient Multi-Vector Retrieval**（arXiv:2605.30120）。

本目录是一个可跑通的 **toy 级** 复现：用稀疏自编码器（SAE）把 token embedding 映射到高维稀疏空间，并在稀疏空间里做 ColBERT-style 的 late-interaction（MaxSim），同时提供一个可工作的倒排索引（inverted index）版本的检索。

> 说明：论文中包含大量系统工程细节（例如 SSR++ 的 coarse-to-fine、CPU/GPU 优化、规模化索引与工程加速、更多损失项/正则等）。本复现聚焦“**稀疏编码替代聚类压缩、并可用倒排索引做 multi-vector 检索**”这一核心思路；未覆盖的系统级优化写在本文末的“未实现部分与伪代码”。

## 目录结构

- `src/ssr_model.py`：Embedding Encoder + Sparse AutoEncoder + MaxSim 打分
- `src/inverted_index.py`：倒排索引构建与基于共享 neuron 的 MaxSim 计算
- `src/toy_data.py`：toy 数据集生成与 DataLoader
- `scripts/generate_toy_data.py`：生成 toy 数据（docs/queries/labels）
- `scripts/train_ssr.py`：训练（重建 + 对比学习）
- `scripts/evaluate.py`：评估 dense vs SSR（Recall@K）并做简单耗时统计

## 环境

```bash
pip install -r requirements.txt
```

## 一键跑通

在 `SSR/` 目录下执行：

```bash
python scripts/generate_toy_data.py --out_dir data/toy
python scripts/train_ssr.py --data_dir data/toy --out_dir data/ckpt
# 默认评估会截断一部分 doc/query，避免 toy 全量 brute-force 太慢
python scripts/evaluate.py --data_dir data/toy --ckpt_dir data/ckpt

# 如需全量跑：
# python scripts/evaluate.py --data_dir data/toy --ckpt_dir data/ckpt --max_docs 2000 --max_queries 500
# 如需额外跑 sparse brute-force（更慢）：
# python scripts/evaluate.py --data_dir data/toy --ckpt_dir data/ckpt --run_sparse_bruteforce
```

你将看到：
- Dense（不加 SAE）与 SSR（稀疏空间）两种检索的 Recall@1/5/10
- 使用倒排索引计算的 SSR Recall@K（toy 规模下与 brute force 一致，速度不作为真实对比）

## 方法概览（对齐论文主线）

1) **Token-level multi-vector 表征（ColBERT-style）**：query 和 doc 都是一串 token embedding。

2) **Sparse AutoEncoder（SAE）做稀疏编码**：
- 编码：`z = TopK(W_enc (x - b_pre) + b_enc)`，只保留 top-k neuron。
- 解码：`x_hat = W_dec z + b_pre`，用重建损失让稀疏空间保留语义。

3) **在稀疏空间里做 late-interaction MaxSim**：
- `score(q,d) = sum_{t in q} max_{s in d} dot(z_q[t], z_d[s])`

4) **倒排索引（inverted index）**：
- 对每个 neuron u，记录 posting list：出现过该 neuron 的 (doc_id, token_id, value)
- query token 的 top-k neuron 很少，只需要遍历这些 neuron 的 posting list，就能枚举“可能相似”的 doc token，从而计算 MaxSim。

## 未实现部分与伪代码（系统工程差异说明）

未实现点：
- SSR++ 的 coarse-to-fine pruning、更多加速策略
- 论文中的更多损失项（multi-topk/aux loss/sparse contrastive 等）仅保留最核心的“重建 + 对比 CE”
- 大规模 BEIR/MSMARCO 真实训练与复现实验

倒排索引计算单个 query token 的 MaxSim（伪代码）：

```text
# q_token: 稀疏向量 (idx_q, val_q)
# postings[u] = list of (doc_id, token_id, val_d)

partial = dict()  # key=(doc_id, token_id) -> dot
for (u, vq) in q_token.nonzero():
    for (doc_id, tok_id, vd) in postings[u]:
        partial[(doc_id, tok_id)] += vq * vd

max_per_doc = dict(doc_id -> -inf)
for ((doc_id, tok_id), dot) in partial.items():
    max_per_doc[doc_id] = max(max_per_doc[doc_id], dot)

return max_per_doc  # 作为该 query token 对每个 doc 的 MaxSim 贡献
```
