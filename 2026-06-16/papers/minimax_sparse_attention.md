# MiniMax Sparse Attention

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | MiniMax Sparse Attention |
| **作者** | Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, Pengyu Zhao |
| **机构** | MiniMax |
| **链接** | https://arxiv.org/abs/2606.13392 |
| **arXiv ID** | 2606.13392 |
| **提交日期** | v1: June 11, v2: June 12, 2026 |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**中文：**  
MiniMax提出了MiniMax Sparse Attention（MSA），一种基于分组查询注意力（GQA）的分块稀疏注意力机制，使前沿LLM能够以可接受的计算代价处理百万级别的超长上下文。MSA设计了一个轻量级**索引分支（Index Branch）**，对Key-Value块进行评分并为每个GQA组独立选择Top-k块，实现组特定的稀疏检索，同时**主分支（Main Branch）**对选中块执行精确块稀疏注意力计算。该方法以简洁性和可扩展性为核心设计原则，可在多种GPU上高效部署。MiniMax M3模型（1M上下文）正是基于此架构。

**English:**  
MiniMax proposes MSA (MiniMax Sparse Attention), a blockwise sparse attention over GQA. A lightweight Index Branch scores KV blocks and selects Top-k per GQA group (group-specific sparse retrieval); the Main Branch performs exact block-sparse attention over selected blocks. Enables 1M-token context at ~1/20 compute vs. full attention. MiniMax M3 (released June 1, 2026) is built on this architecture.

---

## 故事弧线 / Story Arc

**现有方法不足 →** Softmax全注意力的二次计算成本使百万token上下文在部署规模下不可行。  
**本文设计 →** 分块稀疏+轻量索引分支，实现百万token上下文的工业级部署，prefill加速9.7×，decode加速15.6×。

---

## 创新性 / Innovation

1. **组特定稀疏检索**：为每个GQA组独立选择不同的KV块子集，比全局稀疏更精确。
2. **Index Branch + Main Branch解耦**：索引评分与注意力计算分离，模块清晰，可扩展。
3. **工业级验证**：已集成到MiniMax M3（1M上下文），prefill 9.7×、decode 15.6×加速。

---

## 关键指标 / Key Metrics

| 指标 | 结果 |
|------|------|
| 上下文长度 | 1,000,000 tokens |
| 计算量（vs全注意力@1M ctx） | ~1/20 |
| Prefill加速 | 9.7× |
| Decode加速 | 15.6× |
| RULER (1M ctx) / NIAH 召回率 | 接近满分 |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 工程创新强，理论贡献中等 |
| 实验SOTA增量 (SOTA Delta) | 12 | 15 | 大幅降低长上下文计算成本 |
| 实验质量/消融 (Exp Quality) | 12 | 15 | M3生产验证，消融充分 |
| 效率 (Efficiency) | 9 | 10 | 核心贡献即效率 |
| 泛化性 (Generalization) | 4 | 5 | 通用注意力机制 |
| 领域相关性 (Domain Relevance) | 6 | 25 | 通用基础设施，间接相关 |
| **Total** | **65** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.13392  
- GitHub: https://github.com/MiniMax-AI/MiniMax-M3  
- HuggingFace: https://huggingface.co/papers/2606.13392
