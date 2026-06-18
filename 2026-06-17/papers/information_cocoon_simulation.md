# Do Generative Recommenders Deepen the Information Cocoon?

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Do Generative Recommenders Deepen the Information Cocoon? A Closed-Loop Simulation with LLM-powered User Simulators |
| **Authors** | Jiyuan Yang, Gengxin Sun, Mengqi Zhang, Lingjie Wang, Yuanzi Li, Hongxi Cui, Xin Xin, Pengjie Ren |
| **Affiliations** | Shandong University; Renmin University of China |
| **arXiv** | — |
| **Submitted** | 2026-06-17 window |
| **Domain Tags** | information diversity, content cocoon, simulation, LLM agent, generative recommendation |

---

## 方法概述 / Method Summary

生成式推荐器（GenRec）是否比传统推荐器更容易形成"信息茧房"？本文首次构建 LLM-agent 闭环仿真系统，用 LLM 驱动虚拟用户在生成式 vs 传统推荐器下持续交互，记录物品多样性、长尾曝光、头部集中度等指标，并提出原创的"代码空间结构茧房"指标（Code-Space Structural Cocoon Metric）量化语义层面的内容同质化。结论：生成式推荐器在某些场景下确实加剧信息茧房，但可通过多样性干预缓解。

**Story arc**: GenRec 引发多样性监管隐忧，但缺乏系统实验 → LLM-agent 闭环仿真 + 新型茧房指标，首次量化比较 GenRec vs 传统推荐的信息茧房效应。

**Domain relevance**: 内容多样性治理、创作者公平、长尾曝光是电商内容生态核心议题。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 25 | 30 | 首次 LLM-agent 闭环仿真 GenRec 茧房，新型指标 |
| Experimental SOTA delta | 9 | 15 | 诊断性分析，无准确率 SOTA |
| Experimental quality / ablations | 12 | 15 | 多推荐器多场景比较 |
| Efficiency | 6 | 10 | 仿真系统，无效率专项 |
| Generalization | 2 | 5 | 仿真场景局限 |
| Domain relevance | 19 | 25 | 内容多样性与创作者公平，与电商内容生态高度契合 |
| **Total** | **73** | **100** | 议题高度相关，但属诊断性研究，缺乏准确率 SOTA |
