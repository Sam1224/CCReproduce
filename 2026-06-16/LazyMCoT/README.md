# LazyMCoT （复现）

> **Focus When Necessary: Adaptive Routing and Collaborative Grounding for Training-Free Visual Grounding**
> Tencent BAC 等，arXiv **2606.16158**
> 论文：https://arxiv.org/abs/2606.16158ㅤ·ㅤ官方仓库：https://github.com/TencentBAC/LazyMCoT

本目录是 **LazyMCoT** 的一个忠实、可运行的 PyTorch 参考实现。截至复现时，官方仓库仅含 `LICENSE`（无代码），因此这里从论文方法出发完整复现 **流水线结构 + 路由 / 定位逻辑**，并在玩具数据集上端到端跑通（CPU 数分钟）。真实场景里 VLM 与视觉专家（SAM3）可直接替换为真实骨干。

---

## 1. 论文解决的问题

免训练视觉定位方法对**所有**样本无差别地做缩放 / 裁剪 / 重渲染：

- **浪费算力**：统计显示约 **67%** 的样本 VLM 一次前向即可答对，根本不需要额外定位；
- **反伤精度**：无差别的裁剪会截断全局上下文、引入背景噪声，反而把本可答对的简单样本带错。

LazyMCoT 的核心思想是 **"按需聚焦"（Focus When Necessary）**：

1. 用单次前向的 **首答案 token** 抽取两个**零成本**统计量来判断难度；
2. 用 **GBDT 路由器 + 共形预测（conformal）阈值** 决定是否需要定位，并对"难样本召回"给出**可控下界**；
3. 仅对被判为困难的样本启动 **Collaborative Grounding（协同定位）**，精确抠出关键证据后回灌 VLM。

---

## 2. 方法与公式

### 2.1 零成本首 token 统计量

单次前向得到首答案 token 在词表 `V` 上的 logits `z`，候选选项索引集合为 `O`：

- **option top probability**（在选项上重归一化后的最大概率）

  ```
  topp = max_{i∈O} softmax(z|O)_i ,   softmax(z|O)_i = exp(z_i) / Σ_{k∈O} exp(z_k)
  ```

- **option-vs-non-option logit gap**（选项内最大 logit 与选项外最大 logit 之差）

  ```
  delta_logit = max_{i∈O} z_i  −  max_{j∉O} z_j
  ```

二者只需一次前向即可得到（零额外成本）。对应 `model.compute_first_token_stats`。

### 2.2 自适应路由（Adaptive Routing）

在**留出**的路由校准集 `D_cal` 上，用特征 `x=(topp, delta_logit)`、标签 `y∈{0,1}`（`0`=直接答对 / ori-correct，`1`=直接答错 / ori-wrong）**一次性**训练一个 GBDT 概率分类器 `g_θ`，路由分数：

```
p = g_θ(x) ,   s(x) = log( p / (1 − p) )      # 越大表示越可能是"难/答错"样本
```

**共形阈值校准（conformal calibration）**：在 must-recall 子集 `D_mr`（即所有 ori-wrong 样本）上，用**交叉验证的 out-of-fold 分数** `{s_i}` 取经验 α 分位数作为下界：

```
s_floor = Q_α( { s_i : i ∈ D_mr } )
```

这样约 `(1−α)` 比例的难样本满足 `s ≥ s_floor`，即对难样本召回给出**可控下界**；**α 越小 ⇒ s_floor 越低 ⇒ 路由越多 ⇒ 越保守**。

**路由规则**：

```
if s(x) < s_floor :  return Direct(I, Q)      # 单次前向的答案，短路
else              :  return CG(I, Q)          # 进入协同定位
```

对应 `model.AdaptiveRouter`（`fit / score / calibrate_threshold / decide`）。

### 2.3 Collaborative Grounding（Algorithm 1）

```
输入: 图像 I, 问题 Q, 选项 O
1.  E  ← VLM.decompose_entities(Q)                 # 问题 → 实体集
2.  并行两分支:
      B_exp ← SAM3(I, E)                            # 视觉专家分支
      A     ← VLM.cross_modal_attention(I, E)       # 注意力分支, A ∈ R^{T×N}
3.  注意力 → 显著图:
      对每个实体 token 行 reshape 成空间网格 → 高斯平滑(σ) → 归一化
      跨 T 个实体 token 聚合 → 显著图 S → 阈值 τ → 连通域 → 框集 B_att
4.  stage-1 并集:    B1 = B_att ∪ B_exp
5.  stage-2 精修:    for b ∈ B_att 且未被 B_exp 覆盖(IoU<cov_thr):
                        裁剪放大区域 → 再查 SAM3 → 得 δ 框 → 映回原图坐标
                     B2 = B1 ∪ δ
6.  LPD ← render(I, B2)                             # 彩色边框 + 文字图例 + 放大裁剪面板
7.  return VLM.answer_with_lpd(LPD, Q, O)           # 回灌 VLM 得最终答案
```

对应 `model.CollaborativeGrounding`（`_attention_boxes` / `run`）与 `model.render_lpd`（Localized Panel Display, LPD）。

---

## 3. 忠实实现 vs 桩（Stub）

| 组件 | 状态 | 说明 |
|---|---|---|
| 首 token 统计量 `topp / delta_logit` | **忠实** | 严格按公式实现 |
| GBDT 路由 + 共形 α 分位校准 | **忠实** | `sklearn.GradientBoostingClassifier` + OOF 分位数 |
| 注意力 → 显著图（高斯平滑 / 归一化 / 聚合 / 阈值 → 框） | **忠实** | 纯 numpy 实现 |
| 两阶段 并集 / 裁剪精修（IoU 覆盖判定） | **忠实** | `CollaborativeGrounding.run` |
| LPD 渲染（彩色边框 + 图例 + 放大面板） | **忠实** | PIL 绘制 |
| 端到端编排（路由规则） | **忠实** | `LazyMCoT.predict` |
| **VLM**（`first_token_logits / generate_answer / decompose_entities / cross_modal_attention / answer_with_lpd`） | **桩** | 确定性玩具感知器（粗网格下采样 + 颜色证据读取）。真实中包一层 Qwen2.5-VL 等，文件内含 pseudocode |
| **视觉专家 SAM3**（`detect`） | **桩** | 基于饱和度连通域的玩具检测器。真实中包一层 SAM3 / Grounding-DINO，文件内含 pseudocode |

> 桩的设计动机：玩具 VLM 在**粗 token 网格**上读图，**小目标**被下采样淹没 ⇒ 首 token 置信度低 ⇒ 正是路由要识别的"难样本"；LPD 把目标区域放大后再读 ⇒ 答对。这复现了真实高分辨率小目标定位（V\* Bench / HR-Bench）中"按需聚焦"的精度/时延权衡。

更换为真实骨干：替换 `VLMInterface` 与 `VisualExpertSAM3` 的方法实现即可，其它代码（路由、共形、协同定位、编排）无需改动。

---

## 4. 文件结构

```
LazyMCoT/
├── requirements.txt   # torch, numpy, scikit-learn, pillow, joblib
├── data.py            # 玩具数据集 + D_cal/D_test 划分 + 路由特征采集
├── model.py           # 统计量 / 路由器 / 协同定位 / LPD / VLM·SAM3 桩 / 编排
├── train.py           # 训练路由器：采集统计量 → 拟合 GBDT → 共形校准 → 保存
├── test.py            # 评测：精度、短路比例、平均 VLM 前向次数（时延代理）
└── README.md
```

数据集（`data.py`）：384×384 合成"高分辨率"图，深灰背景上散布灰色干扰块，**一个小的彩色目标**；多选问题"目标是什么颜色"。目标越小 ⇒ 越难。`build_splits` 返回 `D_cal`（路由校准）与 `D_test`；`collect_routing_dataset` 跑一遍 VLM 桩产出 `(topp, delta_logit, label)`。

---

## 5. 运行

```bash
pip install -r requirements.txt

# 1) 训练路由器（α 为共形水平，越小越保守）
python train.py --alpha 0.1

# 2) 在测试集评测
python test.py
```

### 典型输出（`--alpha 0.1`）

```
[1/4] Collecting first-token stats over D_cal (n=320) ...
      ori-acc=0.684  #wrong(must-recall)=101/320
[3/4] Fitting GBDT router g_theta on full D_cal ...
      alpha=0.1  s_floor=-2.2322
[4/4] D_cal routing: routed=0.569  hard-recall=1.000

============================================================
LazyMCoT toy evaluation  (n_test=140, alpha=0.1, s_floor=-2.2322)
============================================================
Direct (single-pass) accuracy : 0.671
LazyMCoT accuracy             : 0.986   (delta +0.314)
Short-circuited (direct)      : 0.350
Routed to grounding (CG)      : 0.650
Avg VLM forward passes / sample (latency proxy): 2.95
Grounding fixed wrong->right  : 44
Grounding broke right->wrong  : 0
============================================================
```

### 精度 / 效率权衡（不同 α）

| α | 路由比例 | LazyMCoT 精度 | 平均前向次数 |
|---|---|---|---|
| 0.05 | 0.66 | 0.993 | 2.97 |
| 0.10 | 0.65 | 0.986 | 2.95 |
| 0.20 | 0.60 | 0.950 | 2.80 |
| 0.50 | 0.36 | 0.836 | 2.07 |

可见：**α 越小 ⇒ 路由越多 ⇒ 精度越高、时延越高**；直接基线恒为 0.671。LazyMCoT 通过**短路简单样本**显著降低平均时延，同时**只对难样本定位**带来大幅精度提升——与论文结论一致。

---

## 6. 备注

- 本复现聚焦**流水线结构与路由/定位逻辑**；VLM 与视觉专家是可插拔的真实骨干（论文用 Qwen2.5-VL-7B + SAM3）。
- 全程 CPU 可跑、依赖极简；`python train.py` 后 `python test.py` 均无错误运行。
- 论文：https://arxiv.org/abs/2606.16158ㅤ·ㅤ官方仓库：https://github.com/TencentBAC/LazyMCoT
