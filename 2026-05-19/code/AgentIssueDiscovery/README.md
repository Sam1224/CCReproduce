# AgentIssueDiscovery — Toy Reproduction

本目录是对论文 **"When Rules Fall Short: Agent-Driven Discovery of Emerging Content Issues in Short Video Platforms"** (arXiv:2601.11634, TikTok Inc.) 的可运行复现（toy 级别）。

## 论文核心思想

短视频平台新兴内容问题涌现速度远超人工发现能力。本文提出自动化的三阶段 pipeline：
1. **LLM Agent 召回**：多模态分析视频，识别可能包含新兴问题的候选视频
2. **两阶段聚类**：区分已知问题变体 vs. 全新子问题
3. **策略自动生成**：从聚类簇中生成标注策略描述

## 目录结构

```
AgentIssueDiscovery/
├── agent/              LLM Agent 召回模块
│   ├── base_agent.py   基础 Agent 接口
│   ├── recall_agent.py 多模态视频召回 Agent
│   └── prompts.py      提示词模板
├── clustering/         两阶段聚类
│   ├── embedder.py     视频内容嵌入
│   ├── coarse_cluster.py  粗粒度聚类（HDBSCAN）
│   └── fine_cluster.py    细粒度聚类（区分变体 vs 新问题）
├── policy_gen/         策略生成
│   └── policy_generator.py  从聚类簇自动生成策略描述
├── data/               合成 toy 数据
│   └── synthetic_videos.py
├── pipeline.py         端到端 pipeline
├── eval.py             评测脚本（F1 / Cluster purity）
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行完整 pipeline（toy 数据）
python pipeline.py --num_videos 500 --output_dir runs/toy

# 评测
python eval.py --results_dir runs/toy
```

## 关键指标（toy）

- Issue Discovery F1（vs ground truth）
- Cluster Purity（聚类纯度）
- New Issue Detection Rate（新问题发现率）
