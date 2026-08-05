# Pailitao-MMSearch (toy reproduction)

> 论文：**Pailitao-MMSearch: Building Native E-Commerce Multimodal Search Foundation**
>
> 本 toy 复现的目标不是还原工业级“拍立淘”多模态检索系统的全部工程细节，而是把论文中最关键的“形状”压缩成**最小可运行**的 PyTorch 代码：
>
> 1. **HybSID**：每个 query/item 同时拥有（a）离散语义 ID（Semantic IDs，来自向量量化）与（b）连续向量表示，并在检索打分时混合。
> 2. **两阶段持续预训练 / 蒸馏**：提供 `stage1_pretrain()` 与 `stage2_distill()` 的简化接口。
> 3. **混合推理后训练**：提供 `stage3_posttrain_hybrid_inference()`，冻结 encoder，仅训练混合打分头，使“离散过滤 + 连续 rerank”的推理形态更贴近线上。

目录结构（本任务要求的文件）：

- `data.py`：toy 电商多模态检索数据（文本 + 图像张量 + 目标商品）
- `model.py`：HybSID 双塔、向量量化语义 ID、蒸馏 teacher、混合推理打分头
- `train.py`：训练并保存 checkpoint
- `test.py`：评估 Recall@K
- `requirements.txt`

---

## 1. Toy 数据是什么

我们构造一个**小型商品库（catalog）**：每个 item 有

- `title`：如 `"red shoe sport"`
- `image`：形状 `[3, 32, 32]` 的张量（颜色/品类对应不同的统计模式 + 噪声）

再构造 query：

- `query_text`：与目标 item 的属性高度相关，但会随机缺失/扰动词
- `query_image`：由目标 item 图像 + 噪声得到
- `target_item_id`：目标商品

这样模型可以在 CPU 上快速学到“把 query 对齐到目标 item”的基本检索能力。

---

## 2. HybSID 在本 toy 中如何对应论文

论文中的 HybSID 大意是：

- 用一种离散语义 token/ID 表示（便于倒排/过滤/粗排）
- 同时保留连续向量表示（便于精排、泛化）

本 toy 用 **Group Vector Quantization** 来模拟离散语义 ID：

- 将连续向量 `z ∈ R^D` 切成 `G` 组，每组 `D/G`
- 每组做最近邻量化到 codebook，得到 `sid ∈ [0..K-1]^G`
- 得到量化向量 `z_quant`（并用 straight-through estimator 反传）

检索时的混合打分：

- 连续相似度 `s_cont = cos(zq_cont, zi_cont)`
- 离散相似度 `s_sid = weighted_match_rate(sid_q, sid_i)`
- 最终分数 `s = w_cont*s_cont + w_sid*s_sid`

---

## 3. 训练流程（对应论文形状）

### Stage 1：持续预训练（toy 版）

目标：对齐 query 与 item 的表征（双向 in-batch InfoNCE）。

- 输入：`(query_text, query_image) -> zq`，`(item_text, item_image) -> zi`
- loss：
  - `L_nce(zq_hyb, zi_hyb)`
  - `+ β * L_vq`（量化 commitment loss）

### Stage 2：蒸馏（toy 版）

论文中 teacher 往往是更强的多模态 encoder 或历史版本；这里用一个**固定随机特征**的 `ToyTeacherEncoder` 模拟 teacher 表征，并提供蒸馏接口。

- loss：
  - `L_nce`（保持检索能力）
  - `+ α * (MSE(zq_cont, tq) + MSE(zi_cont, ti))`

### Stage 3：混合推理后训练（toy 版）

线上常见形态：

1. 用离散 ID 进行候选过滤/粗排
2. 用连续向量做 rerank

本 toy 的 stage3 冻结 encoder，仅训练 `HybridScoreHead` 的 `(w_cont, w_sid, sid_group_weight)`，用一个简单的 margin ranking loss 来让最终混合分数更适合检索。

---

## 4. 运行方式

在仓库根目录执行也可以，但推荐进入目录运行：

```bash
cd CCReproduce/2026-08-05/PailitaoMMSearch
pip install -r requirements.txt

# 训练（默认跑 stage1+stage2+stage3，并保存 ckpt）
python train.py --ckpt_path checkpoints/pailitao_mmsearch.pt

# 测试 Recall@K
python test.py --ckpt_path checkpoints/pailitao_mmsearch.pt
```

可选参数示例：

```bash
python train.py --num_items 300 --num_queries 800 --epochs_stage1 3 --epochs_stage2 2 --epochs_stage3 1
python test.py --candidate_m 50 --ks 1 5 10
```

---

## 5. 与论文真实系统的差距（必须说明）

本 toy **无法完整实现**论文中的工业级细节，包括但不限于：

1. **大规模语义 ID 生成**：论文可能包含更复杂的 semantic tokenizer / transformer / 多粒度离散码；此处用 VQ 近似。
2. **两阶段持续预训练**：真实系统可能包含多任务、多域数据、长周期增量训练与复杂调度；此处仅保留接口与最小 loss。
3. **蒸馏方式**：真实 teacher 可能是更强模型（更大 backbone、跨模态交互、长序列等）；此处 teacher 为固定随机特征模型，只用于演示接口。
4. **检索工程**：论文中通常有 ANN、倒排、分层召回、在线索引更新、query rewriting 等；此处用纯 PyTorch 张量暴力计算 + 简易 SID 过滤。

### 对应伪代码（更贴近论文但未实现）

```text
# (Pseudo) industrial-scale HybSID retrieval
sid_q = semantic_tokenizer(query)
cont_q = encoder(query)

C = inverted_index.lookup(sid_q)           # coarse candidates by discrete IDs
C = rerank_by_continuous(C, cont_q)        # dense reranking
C = postprocess(C)                          # business rules, diversity, etc.
return topK(C)
```

代码中所有“未实现的真实细节”也在对应模块 docstring 中再次说明。
