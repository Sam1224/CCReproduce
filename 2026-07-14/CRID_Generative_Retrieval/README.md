# CRID Generative Retrieval（toy reproduction）

复现论文：**Beyond Semantic IDs: Encoding Business-Value Ranking into Document Identifiers for Generative Retrieval** 的一个 *toy-but-runnable* CPU 版本流水线。

本 toy 的目标是用最小代码体现论文核心直觉：

- **Semantic ID (SID)**：DocID 由“语义聚类前缀” + “簇内随机后缀”组成（不包含业务价值排序信息）。
- **CRID**：DocID 由“语义聚类前缀” + “簇内业务价值 rank 后缀”组成（rank 越小业务价值越高）。
- 在 **Generative Retrieval** 框架下，模型从 Query 直接 **生成 DocID token**；当模型对簇内精确 DocID 不确定时，CRID 能把这种不确定性“自然地”转化为**更高业务价值的排序**（因为低 rank token 在全局上可共享、可泛化）。

> 这不是论文的完整工业系统复现，而是把关键机制压缩成 CPU 可跑通的小实验。

---

## 目录结构

- `data.py`：自生成 toy 数据（文档、查询、业务价值），做一次“语义聚类”，并构造 SID/CRID 两套 DocID。
- `model.py`：一个极简的两步自回归 DocID 生成模型：先生成聚类前缀 token，再生成簇内后缀 token。
- `train.py`：训练并保存 checkpoint；同时保存数据 artifact。
- `test.py`：加载 checkpoint，在测试集上输出：
  - HR@1/5/10
  - MRR@10
  - MeanValue@5
  - ValueNDCG@10
  并支持 `--id_scheme {crid,sid}` 对比。

---

## 环境

```bash
pip install -r requirements.txt
```

---

## 一键跑通（CPU）

### 1) 训练（CRID）

```bash
python train.py --id_scheme crid
```

### 2) 测试（CRID）

```bash
# 默认 --decode_mode prefix_order：只生成语义簇 prefix，然后利用 DocID 后缀的“自然顺序”做排序
python test.py --id_scheme crid

# 如果你想看“纯生成式”效果（对(prefix,suffix)做 beam search）
python test.py --id_scheme crid --decode_mode beam
```

### 3) 训练/测试（SID）对比

```bash
python train.py --id_scheme sid
python test.py --id_scheme sid
```

默认超参下数据量和模型都很小，CPU 一般数分钟内可跑完。

---

## 你应该看到什么现象？

在本 toy 设定中：

- Query 主要决定“语义簇”（相关主题），
- 业务侧希望在语义相关的前提下，**优先返回更高 business value 的文档**。

由于：

- **CRID 的后缀 token 与 value rank 对齐**（例如 `R00` 表示簇内价值最高）。因此在 `--decode_mode prefix_order` 下，只要模型能生成正确的语义簇 prefix，就可以按 `R00,R01,...` 的自然顺序直接得到“簇内高价值优先”的排序。
- **SID 的后缀 token 与 value 无关且簇内随机**。在相同的 `prefix_order` 解码下，后缀顺序基本等价于随机排序，MeanValue/NDCG 会明显更差。

因此通常会观察到：

- HR/MRR（是否命中语义相关簇）两者接近；
- **MeanValue@5、ValueNDCG@10：CRID 显著高于 SID**。

---

## 与论文实现的主要差异（简化点）

- 只做了 **两段式 DocID**（prefix + suffix），没有做层次化多级语义 ID。
- “生成式检索模型”用一个小 GRU encoder + 两个分类 head 近似（两步自回归生成 2 个 token）。
- 数据为自生成 bag-of-words 文本；语义聚类用 numpy k-means 实现。
- 训练目标是生成一个正样本文档的 DocID（通过 value-biased 采样体现“业务偏好”），而非完整的工业检索训练范式。

---

## 产物（artifact）

默认会在当前目录下创建：

- `artifacts/toy_data.pt`：文档、查询、语义簇、业务价值、SID/CRID 映射等。
- `runs/{id_scheme}/ckpt.pt`：模型参数 + tokenizers + 训练配置。

