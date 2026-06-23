# OneBar：面向电商视频流的端到端内容锚定生成式 Query 推荐框架（非官方复现）

> **论文**：OneBar: An End-to-End Content-Grounded Generative Query Recommendation Framework for E-Commerce Video Feeds
> **arXiv**：https://arxiv.org/abs/2606.15330
> **机构**：浙江大学（Zhejiang University）、快手科技（Kuaishou Technology）
> **本目录**：`2026-06-23/OneBar/` — 基于 PyTorch + HuggingFace transformers 的**忠实、从零复现**。

⚠️ **声明**：本仓库为**非官方复现**。原论文**无官方开源代码**，本实现完全依据论文正文（Sec. 3 方法、Algorithm 1、Eq. 4–16、Table 1–6）从零编写。生产级组件（真实多模态视频编码器、基于亿级日志的 ANN 召回、在线 A/B 系统）无法在本环境复现，相关部分以**带详细伪代码/公式注释的 stub** 给出（见下文“近似与 stub”），绝不静默留空或破坏运行。

---

## 1. 方法概述

OneBar 用**一个端到端生成模型**替代传统多阶段级联（MCA）的短视频底栏（Bottom-Bar）Query 推荐，满足 20–30ms 的在线时延预算。三大核心组件：

### (1) Collaborative-Multimodal Intent Grounding（协同-多模态意图锚定，Sec. 3.2）
将每次视频曝光抽象为结构化证据 schema：

```
E(x,u) = < T_x , M_x , A_x , H_u >
```

| 字段 | 含义 | 本复现实现 |
|------|------|-----------|
| `T_x` | 清洗后的文本标题（去 @提及 / #话题 / 营销词） | `data.clean_title`（正则清洗，真实可运行） |
| `M_x` | 多模态视频摘要（objects/scene/ASR/creator） | 模板合成；真实流程见 `stub_multimodal_summarizer` |
| `A_x` | 协同 Query 锚点（行为对齐 ANN：q→v / v→q / q→q，Table 1） | 同类目采样；真实流程见 `stub_behavior_aligned_ann`（含 Qwen3-Emb 0.6B 阶段化训练、相似度 >0.88 过滤伪代码） |
| `H_u` | 相关性过滤后的用户历史 | 同类目采样（与 trigger 相关而非长期兴趣） |

### (2) Low-Latency Unified End-to-End Generation（统一端到端生成 + 弃权，Sec. 3.3）
将稀疏的四个字段序列化为**紧凑 [SEP] 分隔 schema**（Eq. 4，**Prompt 压缩**）：

```
s_x = [ T_x ; [SEP] ; M_x ; [SEP] ; A_x ; [SEP] ; H_u ]
```

送入 **BART encoder–decoder**（默认 `facebook/bart-base`）：`p_θ(y|s_x)=∏_t p_θ(y_t|y_<t, Enc_θ(s_x))`，beam width=8 生成 **K=8** 个有序候选；输出可为正常 query 或弃权符 `[REJECT]`。

> **关键消融（Table 4）**：字段对齐的压缩 prompt（HR@8 **0.3564**）远超冗长指令式 prompt（HR@8 **0.1864**）。`data.build_prompt(style="compressed"|"verbose")` 同时实现了两种，可直接对比。

### (3) Progressive Preference Internalization（渐进式偏好内化，Sec. 3.4）
三阶段训练：

- **Stage 1 — Content-Grounded SFT**（Eq. 5）：在点击 query `y+` 上做交叉熵；非合规 trigger 监督为 `[REJECT]`（学习“何时弃权”）。
- **Stage 2 — List-wise Feedback Alignment**（Eq. 6–10）：基于 6 级行为偏好列表（Table 2）的 **list-wise Softmax DPO**，并做**单调 CTR 过滤**。
- **Stage 3 — PIOPD**（Eq. 11–16，Algorithm 1）：**无需单独训练 reward model**。冻结 teacher 接收后验增强 prompt `x^(T)=x⊕_rand y_ref`（随机位置插入点击 query），student 仅见 `x^(S)=x`；对 student 的**在策略 rollout** 用 **FKL + RKL** 蒸馏 teacher 软分布，并保留 **SFT 锚点** + **熵正则**（抗 top-1 坍塌，Fig. 4）+ 可选 **R-Drop / FGM**。

6 级行为层级（Table 2，1 最强）：

```
1 当前用户点击且有点后消费   > 2 当前用户点击
3 他人点击且有点后消费       > 4 他人点击
5 曝光未点击                 > 6 召回未曝光
```

---

## 2. 复现架构（代码结构）

```
2026-06-23/OneBar/
├── README.md            # 本文件
├── requirements.txt     # torch / transformers / numpy / sacrebleu
├── data.py              # 合成数据生成器 + 证据 schema 序列化/压缩 + 3 个 Dataset
├── model.py             # OneBarGenerator：BART 封装（生成 K=8、teacher/student logits、on-policy 采样）
├── losses.py            # SFT-CE / list-wise DPO / PIOPD(FKL+RKL+SFT+Ent+R-Drop)
├── train_sft.py         # Stage 1
├── train_listwise.py    # Stage 2（含冻结 reference）
├── train_piopd.py       # Stage 3（含冻结 teacher、随机后验 prompt、FGM）
└── evaluate.py          # Exact HR@8 / MRR@8 / ED-HR@8 / BLEU@8 / BAS@8(占位)
```

核心公式与代码对应：

| 论文 | 代码位置 |
|------|---------|
| Eq. 4 序列化/压缩 | `data.build_prompt` |
| Eq. 5 SFT CE | `losses.sft_loss` |
| Eq. 6–10 list-wise DPO | `losses.listwise_dpo_loss`（`r_i=ℓ_θ(y_i)-ℓ_ref(y_i)`，clicked anchor `b(y_j)≤4`） |
| Eq. 11 随机后验 prompt | `data.PIOPDDataset._teacher_prompt` |
| Eq. 13 FKL+RKL 蒸馏 | `losses.distill_loss`（`τ²·[λ_fkl·KL(p_T‖p_S)+λ_rkl·KL(p_S‖p_T)]`，τ=2） |
| Eq. 14 熵正则 | `losses.entropy_regularizer` |
| Eq. 15 R-Drop / FGM | `losses.rdrop_loss` / `train_piopd.fgm_attack` |
| Eq. 16 PIOPD 合成 | `losses.piopd_loss` |
| Algorithm 1 | `train_piopd.main` |

---

## 3. 如何运行

### 安装
```bash
pip install -r requirements.txt
# 如需代理：
# export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
# export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
```

### 生成玩具数据（可选，Dataset 也会按需即时生成）
```bash
python data.py --n 64 --out toy_data.jsonl
```

### 三阶段训练（CPU 冒烟，几步即可）
```bash
# Stage 1: SFT
python train_sft.py      --tiny --max_steps 3 --batch_size 2 --save_dir outputs/sft
# Stage 2: list-wise DPO（从 Stage1 初始化）
python train_listwise.py --init_from outputs/sft      --max_steps 3 --save_dir outputs/listwise
# Stage 3: PIOPD（从 Stage2 初始化，可加 --lambda_rdrop / --lambda_fgm）
python train_piopd.py    --init_from outputs/listwise --max_steps 3 --lambda_rdrop 0.5 --lambda_fgm 0.5 --save_dir outputs/piopd
```

### 评测
```bash
python evaluate.py --init_from outputs/piopd --n_eval 24
```

### 复现完整设置（需 GPU / 真实数据）
- 去掉 `--tiny`，默认使用 `facebook/bart-base`。
- 论文超参：Stage1 lr=1e-5、bs=128、6 epoch；Stage2 lr=5e-5、β=0.1；Stage3 lr=1e-6、bs=64、500 步、τ=2。
- 用 `--prompt_style verbose` 复现 Table 4 的 prompt 压缩消融。

---

## 4. 评测指标（Sec. 5.1）

`evaluate.py` 实现：
- **Exact HR@8**：top-8 中存在与 GT 完全一致的 query 即命中。
- **MRR@8**：首个精确命中的倒数排名均值。
- **ED-HR@8**：Levenshtein 距离 ≤2 即命中（容错语义等价表述）。
- **BLEU@8**：top-8 中最高的 sentence-BLEU（`sacrebleu`；缺失时退化为 char-ngram 代理）。
- **BAS@8（占位）**：论文用协同数据训练的 embedding 算 trigger–query 相似度；本复现用 char-ngram 余弦近似，并在输出中明确标注 `(placeholder)`。替换 `evaluate._char_ngram_cosine` 为真实 embedding 余弦即为忠实实现。

---

## 5. 近似与 stub（透明说明）

| 组件 | 论文做法 | 本复现 | 位置 |
|------|---------|--------|------|
| 多模态摘要 `M_x` | 关键帧+ASR 经 MLLM 摘要，离线日更 | 模板合成（接口一致） | `data.stub_multimodal_summarizer`（含伪代码） |
| ANN 协同锚点 `A_x` | Qwen3-Emb 0.6B 阶段化行为监督，128 维空间，sim>0.88 过滤 | 同类目采样 | `data.stub_behavior_aligned_ann`（含伪代码） |
| 离线日志/CTR | 真实曝光点击日志 | 6 级模板合成 + 单调 CTR 过滤（真实算法） | `data._build_preference_list` / `monotonic_ctr_filter` |
| BAS@8 embedding | 协同 embedding 余弦 | char-ngram 余弦占位 | `evaluate._char_ngram_cosine` |
| 离线权重下载 | `facebook/bart-base` | 下载失败时**自动回退**到随机初始化的 tiny BART，保证离线可运行 | `model.OneBarGenerator._build_backbone` |

> 除上述明确标注外，**绝大部分逻辑均为真实可运行实现**：文本清洗、prompt 压缩/序列化、SFT、list-wise Softmax DPO（含 log-ratio 与 clicked-anchor 选择）、PIOPD 全流程（在策略 rollout、teacher/student 双 KL、SFT 锚点、熵正则、R-Drop、FGM）、以及全部评测指标。

---

## 6. 离线回退说明

为保证无网络环境可运行：若 `facebook/bart-base` 权重下载失败，`OneBarGenerator` 会自动回退到一个**小型随机初始化 BART config**（`model._tiny_config`，2 层、d_model=128），仅用于跑通流程；冒烟测试的指标因此接近 0，属预期现象。生产复现请使用 `facebook/bart-base` 全量权重并在真实数据上训练。
