# Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer is Key to Unification |
| **Authors** | Wujian Peng, Lingchen Meng, Yuxuan Cai, Xianwei Zhuang, Yuhuan Yang, Rongyao Fang, Chenfei Wu, Junyang Lin, Zuxuan Wu, Shuai Bai |
| **Affiliations** | Fudan University; Shanghai Innovation Institute; Qwen Team, Alibaba Inc. |
| **arXiv** | — |
| **Submitted** | 2026-06-17 window (accepted ICML 2026) |
| **Domain Tags** | multimodal, autoregressive, tokenizer, generation, understanding, Qwen |

---

## 方法概述 / Method Summary

统一多模态理解与生成是当前的核心挑战。本文提出共享上下文-视觉 Tokenizer（Shared Context-Visual Tokenizer）作为统一的关键：使用单一离散 Tokenizer 编码图像/视频（而非分别用理解 vs 生成专属 Tokenizer），结合二值量化（Binary Quantization）和并行比特自回归（Parallel Bitwise Autoregression）实现高效的图像生成、理解和文本渲染。模型在生成质量、编辑和文本渲染多项超越 GPT-4o，同时保持计算高效。

**Story arc**: 统一多模态模型的核心瓶颈是"理解"与"生成"使用不同 Tokenizer 导致模态鸿沟 → 设计共享离散 Tokenizer + 并行比特自回归，统一两阶段任务。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 27 | 30 | 共享离散 Tokenizer + 二值量化 + 并行比特自回归 |
| Experimental SOTA delta | 12 | 15 | 多项超 GPT-4o，ICML 2026 |
| Experimental quality / ablations | 13 | 15 | 理解/生成/文本渲染全面评估 |
| Efficiency | 9 | 10 | 并行自回归加速 |
| Generalization | 4 | 5 | 多模态多任务 |
| Domain relevance | 9 | 25 | 与电商内容生态间接相关（图文生成、商品图编辑） |
| **Total** | **74** | **100** | 高热度统一多模态论文，但与创作者治理相关性间接 |
