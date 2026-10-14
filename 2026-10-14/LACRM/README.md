# LACRM toy reproduction

- **Paper**: *LACRM: LLM-Guided Mining of Directed Advertiser Competition Relationships in E-Commerce Advertising*
- **Authors**: Xiang Shi, Yitian Kou, Bing Li, Shengsheng Niu, Chun Gan, Linghui Meng, Jie He, Zhangang Lin, Ching Law
- **Affiliations**: JD.com; East China Normal University
- **arXiv**: https://arxiv.org/abs/2610.13959
- **Reproduction score**: 80/100 · status: *reproduced* (no usable public code found in today’s public check)

这是一个 **toy-but-runnable** 的 PyTorch 复现，目标不是复刻私有工业依赖，而是把论文的主链路尽可能完整地跑通：

1. **LLM-Guided Dual-Profile Generation**
2. **Asymmetric Embedding and Multi-Route Retrieval**
3. **Dynamic Behavioral Alignment**

## 和原论文的一一对应

| 论文模块 | 本复现实现 |
| --- | --- |
| AR / QR 双画像 | `data.py` 中的结构化 advertiser fields + `model.py` 中的 `DualProfileEncoder` |
| 5 个 query views | `VIEW_NAMES` + `VIEW_FIELD_MASKS` |
| 视角路由检索 | `train.py::build_candidate_bank` / `test.py` |
| 行为特征 | `data.py::make_behavior_features` |
| LambdaMART list-wise 排序 | `model.py::LambdaRankReranker` + `train.py::lambdarank_loss`（作为 semantic score 的 residual refinement） |
| P@K / R@K / NDCG@K | `test.py` |

## 真实论文中未直接公开、因此做了可运行替代的部分

### 1) Profile generation LLM

原论文使用：

- `Qwen3-4B-Instruct`
- teacher-generated profiles
- SFT + auxiliary GRPO

本复现中，改为：

- 根据商家结构化字段模板化生成 AR / QR；
- 再用 view-conditioned dual encoder 去学习“哪些字段在某个 view 下更重要”。

代码中保留了这一替代的接口，因此如果未来有可用的 profile generator，只需要替换输入文本或 profile tokenizer。

### 2) Embedding backbone

原论文使用：

- `Qwen3-Embedding-8B`
- 不额外训练 embedding model

本复现中，改为：

- PyTorch 字段嵌入层 + source/target 角色投影；
- 用 train-window 的 realized edges 做轻量 InfoNCE 式对比训练。

这不是逐字复刻，但仍然保持了：

- **anchor-side AR**
- **query-side QR**
- **instruction / view-conditioned asymmetry**
- **per-view ANN-like retrieval**

### 3) LambdaMART

原文使用 GBDT 风格的 `LambdaMART`。

由于这里要求 PyTorch 可运行复现，因此改成了一个 **LambdaRank-style differentiable reranker**：

- 输入仍然是 `5 semantic scores + 5 hit indicators + behavioral features`
- 训练目标仍然是 list-wise ranking
- 输出仍然是按 source merchant 分组的 directed competition ranking

## Toy 数据集设计

我们构造了一批合成商家，每个商家包含论文中对应的关键字段：

- `f1` Category & Brand
- `f2` Search Intent
- `f3` Style & Positioning
- `f4` Audience & Scenario
- `f5` Price Band
- `f6` Shop Identity / Channel

此外还额外构造：

- merchant strength / tier（制造竞争方向性）
- festival activity（决定 temporal view 是否激活）
- 6 组行为特征（traffic / conversion / dynamics / lifecycle / quality / intra-category trend）

ground-truth directed competition score 由 5 个 view 的隐式分数与方向性因子共同生成，从而使：

- 同品牌跨类竞争；
- 同品牌同类跨渠道摩擦；
- 同类直接竞争；
- 跨类功能替代；
- 节日活动冲突；

都能在 toy 世界里真实存在。

## 默认结果（CPU，可在 1~2 分钟内跑完）

典型输出会表现出：

- semantic retrieval 已能召回大部分 direct / structural rivals；
- reranker 融合 behavior features 后，`Precision@10 / Recall@10 / NDCG@10` 继续提升；
- 由于 toy 世界中存在 train / test 漂移，evaluation week 中会出现一部分“新兴竞争边”。

一次典型运行的结果会保存在：

- `train_metrics.json`
- `test_metrics.json`

## 运行方式

```bash
cd 2026-10-14/LACRM
pip install -r requirements.txt
python train.py
python test.py
```

## 文件说明

- `requirements.txt` — 依赖（torch、numpy）
- `data.py` — 合成市场、字段、行为特征、历史/评测 rivalry 标签
- `model.py` — dual encoder + LambdaRank-style reranker
- `train.py` — 检索训练、候选生成、list-wise reranker 训练
- `test.py` — semantic-only vs full reranker 对比评测
