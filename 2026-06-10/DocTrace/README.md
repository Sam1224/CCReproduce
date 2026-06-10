# DocTrace (Toy Reproduction)

- Paper: **DocTrace: Tracing Evidence Selection in Long Document QA with Multi-Agent Systems**
- arXiv: https://arxiv.org/abs/2606.10921
- Project page: https://zaixjun.github.io/DocTrace/

## What’s implemented

这是一个 **toy but runnable** 的复现，聚焦论文想解决的两个核心点：

1. **长文档 QA 的证据选择**：将文档拆成段落，先检索再筛选，尽量只把“真正相关的证据”交给回答模块。
2. **Traceable evidence flow**：在检索、筛选、聚合、回答四个阶段输出结构化 trace（证据段落 ID、分数、理由、最终使用的上下文 token 成本），用于诊断与对比。

toy 任务用可控的合成长文档 QA 数据集：每个文档包含大量“相似但错误”的干扰事实，baseline 容易被噪声误导；DocTrace-style 的证据筛选会更稳，同时显著降低上下文 token。

## Limitations (vs. the full paper)

- 原论文使用真实长文档 QA 数据与 LLM multi-agent 系统；这里用规则化合成数据 + 轻量 torch 证据筛选器模拟“可追踪证据选择”的机制。
- 指标仅用于验证机制闭环（Accuracy / token cost），不代表原论文在公开基准上的绝对数值。

## Quickstart

```bash
cd 2026-06-10/DocTrace
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 生成合成数据（长文档 + 噪声）
python generate_data.py --out_dir data/demo

# 2) 训练证据筛选器（Filter Agent 的替身）
python train.py --data_dir data/demo --out_dir runs/demo

# 3) 对比 baseline vs DocTrace-style pipeline
python test.py --data_dir data/demo --ckpt_dir runs/demo
```

运行 `test.py` 会输出两组结果：
- Baseline（TopK-RAG）：直接把检索 TopK 段落拼接用于回答
- DocTrace-style：检索 TopK → 证据筛选 → 聚合 → 回答，并输出 trace 与 token cost
