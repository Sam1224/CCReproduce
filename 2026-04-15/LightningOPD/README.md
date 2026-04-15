# Lightning OPD（Toy Reproduce）

- Paper: **Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation**
- arXiv: https://arxiv.org/abs/2604.13010

本复现为**教学用 toy 版本**，目标是把论文的关键流程（SFT → 预计算 teacher log-prob → Offline OPD 训练）跑通，并演示 **teacher consistency** 对 offline OPD 是否工作具有决定性影响。

## 复现范围（与原论文的差异）

- 原论文面向大型推理模型（Qwen3 等）与真实数学/代码基准；本复现使用一个小型 GRU 语言模型，在合成的加法任务（`a + b`）上训练。
- 原论文的工程细节（大规模 rollout、并行推理、真实 tokenizer、混合精度/分布式等）不在本复现范围内。

## 环境

- Python 3.10+
- PyTorch 2.x（CPU 也可跑通）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1) SFT（teacher 生成监督数据）

```bash
python -m lightning_opd.train_sft --out_dir runs/sft_consistent --teacher_id A
```

### 2) 预计算 teacher log-prob（Offline OPD 的 preprocessing）

```bash
python -m lightning_opd.preprocess_teacher_logits \
  --sft_dir runs/sft_consistent \
  --out_file runs/opd_dataset_consistent.pt \
  --teacher_id A
```

### 3) Lightning OPD（Offline OPD 训练）

```bash
python -m lightning_opd.train_lightning_opd \
  --sft_dir runs/sft_consistent \
  --opd_dataset runs/opd_dataset_consistent.pt \
  --out_dir runs/lightning_opd_consistent
```

### 4) 评测

```bash
python -m lightning_opd.eval \
  --ckpt runs/lightning_opd_consistent/model.pt
```

## Teacher Consistency Demo

用“teacher 不一致”的方式构造 SFT 数据（SFT 用 teacher A，OPD 用 teacher B），观察 offline OPD 会明显更难提升。

```bash
python -m lightning_opd.train_sft --out_dir runs/sft_inconsistent --teacher_id A
python -m lightning_opd.preprocess_teacher_logits --sft_dir runs/sft_inconsistent --out_file runs/opd_dataset_inconsistent.pt --teacher_id B
python -m lightning_opd.train_lightning_opd --sft_dir runs/sft_inconsistent --opd_dataset runs/opd_dataset_inconsistent.pt --out_dir runs/lightning_opd_inconsistent
python -m lightning_opd.eval --ckpt runs/lightning_opd_inconsistent/model.pt
```

你应该能看到 consistent 组整体更稳定、更容易收敛到高准确率。
