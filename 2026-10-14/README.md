# 2026-10-14 电商内容生态 & 达人治理 Paper 巡检

## 今日结论

- 今日候选中，**LACRM** 是唯一达到 `>=80` 复现阈值且未检到可用公开官方代码的论文。
- 已在 `2026-10-14/LACRM/` 下补充 **toy-but-runnable** 的 PyTorch 复现，覆盖：
  - 数据构造
  - 双画像 / 多视角检索
  - 行为特征融合
  - list-wise 排序训练
  - 评测脚本

## 评分机制（百分制）

- 方法创新性：30
- 实验指标：15
- 实验质量：15
- 方法效率：10
- 方法泛化性：5
- 论文相关性：25

## 复现说明

原论文使用了：

1. `Qwen3-4B-Instruct` + teacher profiles 做 AR / QR profile generation；
2. `Qwen3-Embedding-8B` 做 instruction-conditioned asymmetric embedding；
3. `LambdaMART` 做行为对齐重排。

本仓库中的 PyTorch 复现保持了 **方法形状与接口一致**，但对无法公开复现的工业依赖做了替代：

- 用 **模板化 profile 生成 + 可学习字段编码器** 代替私有 LLM profile generator；
- 用 **view-conditioned PyTorch dual encoder** 代替私有 embedding backbone；
- 用 **LambdaRank-style PyTorch reranker** 代替 LambdaMART。

这使得代码可以在 CPU 上直接运行，同时保留论文的主干流程：

`Advertiser / Query dual profiles -> 5-view retrieval -> behavioral alignment reranking`

## 目录

- `papers.json`：当天结构化元数据
- `LACRM/`：代码与说明

