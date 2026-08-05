# MMRM（toy reproduction）

本目录提供一个**最小可运行**的 PyTorch toy reproduction，用于复现论文：

> **MMRM: A Multiplex Multimodal Representation Model for Product Ranking in E-commerce Search**

目标：用尽量少的代码，保留论文的“核心形状（core shape）”，并提供可训练、可评测（NDCG/AUC）的玩具电商搜索排序任务。

---

## 1. 复现保留了什么（What is preserved）

- **共享多模态骨干（shared multimodal backbone）**：同一套编码器同时用于历史商品与候选商品（文本 + 图片特征）。
- **任务 token（task tokens）**：不同任务（click / purchase）用不同的 task token，通过与 multiplex token 的相似度得到任务特定的 multiplex mixing 权重。
- **多路 multiplex item representations**：每个商品从共享 backbone 得到一个 shared embedding，再生成 `K` 路 multiplex 表示（对应论文中“多路表征/多路语义槽位”的概念）。
- **基于行为序列的 multiplex user representations**：对用户历史行为序列，针对每一路 multiplex 表示分别做 attention pooling，得到 `K` 路 user representations。
- **多任务 ranking**：同一套 item/user multiplex 表示上，做 click 与 purchase 两个任务的打分与联合训练。

---

## 2. 简化了什么（What is simplified / not implemented）

该 toy 复现刻意做了很多工程与算法简化，**无法完整实现的部分在代码 docstring 中给出解释与伪代码**：

- 真实论文通常使用大规模预训练多模态骨干（如文本 Transformer + 图像 ViT/ResNet 等），这里用 **embedding + mean pooling（文本）** 和 **线性层（图片）**。
- 论文中的 multiplex 模块在工业系统中可能是更复杂的 cross-attention / transformer / routing（甚至含门控、蒸馏、约束），这里用一个**“item shared embedding + multiplex token → MLP”**的轻量实现来生成 `K` 路表示。
- 未实现线上特征/多塔召回/曝光偏差校正/复杂的 listwise 或 pairwise ranking loss；这里用固定候选集合上的 **pointwise BCE**。
- toy 数据为合成数据：用潜在语义向量生成商品的文本 token 与图片特征，并用（query,user,item）潜在相关性生成 click/purchase 标签。

---

## 3. 代码结构

- `data.py`：合成电商搜索排序数据（query、候选 items、历史行为序列、click/purchase 标签）。
- `model.py`：MMRM 的最小结构：共享多模态骨干 + multiplex item/user 表示 + task token 多任务打分。
- `train.py`：训练并保存 checkpoint（默认 `mmrm_toy.pt`），并在训练结束后打印测试集 NDCG/AUC。
- `test.py`：加载 checkpoint，输出 NDCG@10 与 click AUC。

---

## 4. 运行方式

在本目录下执行：

```bash
pip install -r requirements.txt

python train.py --epochs 5 --batch_size 64 --checkpoint mmrm_toy.pt
python test.py --checkpoint mmrm_toy.pt
```

可选参数：
- `--seed` 控制合成数据与初始化的确定性。
- `--num_candidates` 控制每个 query 的候选集合大小。

---

## 5. 你应该看到什么

- `train.py` 会打印每个 epoch 的 loss，以及训练集/测试集的简要指标。
- `test.py` 至少会输出：
  - `ndcg@10_click`
  - `ndcg@10_purchase`
  - `auc_click`

---

## 6. 与论文实现的对应关系（high-level mapping）

- Shared multimodal backbone → `SharedMultimodalBackbone`
- Multiplex item representations → `MultiplexItemEncoder`（输出形状 `(..., K, D)`）
- Multiplex user representations from behavior sequence → `MultiplexUserEncoder`（输出形状 `(B, K, D)`）
- Task token & multi-task ranking head → `MMRM.task_tokens` + `MMRM.task_mixing_weights()` + `MMRM.rank_logits()`

---

## 7. 伪代码（对应论文更“完整”的 mux/routing 形态，供对照）

> 下述是“更接近论文/工业系统”的写法示意，本 toy 版本用 MLP/attention pooling 做了替代。

```text
# item side
x_shared = SharedBackbone(text, image)
for k in 1..K:
    item_rep[k] = MultiplexBlock(x_shared, mux_token[k])

# user side
for k in 1..K:
    user_rep[k] = AttentionPool(query, history_item_rep[:, k, :])

# multi-task scoring
for task t in {click, purchase}:
    alpha_t = softmax( sim(task_token[t], mux_token[1..K]) )
    score_t = sum_k alpha_t[k] * Match(user_rep[k], item_rep[k], query)
```
