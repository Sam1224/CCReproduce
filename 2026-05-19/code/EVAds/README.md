# EVAds — Toy Reproduction

本目录是对论文 **"E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs"** (arXiv:2602.08355, Alibaba/Taobao) 的可运行复现（toy 级别）。

## 论文核心思想

1. **E-VAds 基准**：首个专为电商短视频设计的 MLLM 评测数据集（3961 条淘宝视频 + 19785 个 Q&A）
2. **信息密度框架**：量化评估电商视频的多模态信息密度（显著高于通用数据集）
3. **E-VAds-R1**：多智能体标注 + RL 微调，用少量样本大幅提升商业意图推理（+109.2%）

## 目录结构

```
EVAds/
├── data/
│   ├── synthetic_evads.py    合成电商短视频 Q&A 数据集
│   └── density_metrics.py    信息密度量化框架
├── model/
│   ├── mllm_wrapper.py       MLLM 微调封装接口
│   └── evads_r1.py           E-VAds-R1 模型（RL 微调框架）
├── annotation/
│   └── multi_agent_annotator.py  多智能体自动标注 pipeline
├── train_rl.py               强化学习微调脚本
├── eval.py                   E-VAds 评测脚本
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 生成 toy 数据集
python data/synthetic_evads.py

# 评测基础 MLLM（toy）
python eval.py --model baseline --num_samples 100

# RL 微调（toy）
python train_rl.py --epochs 3 --output_dir runs/evads_r1

# 评测 E-VAds-R1
python eval.py --model evads_r1 --ckpt runs/evads_r1/best.pt
```

## 关键指标（toy）

- 商业意图推理准确率（Commercial Intent Accuracy）
- 视频问答正确率（VQA Accuracy）
- 信息密度分数（Multimodal Information Density Score）
