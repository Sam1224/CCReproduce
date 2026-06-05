# Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **arXiv** | https://arxiv.org/abs/2601.17058 |
| **GitHub** | https://github.com/weAIDB/awesome-data-llm |
| **提交日期** | 2026-01-22（catch-up 发现；不在 2026-06-04 主窗口，但数据质量主题高度契合） |
| **作者** | Wei Zhou, Jun Zhou, Haoyu Wang, Zhenghao Li, Qikang He, Shaokun Han, Guoliang Li, Xuanhe Zhou, Yeye He, Chunwei Liu, Zirui Tang, Bin Wang, Shen Tang, Kai Zuo, Yuyu Luo, Zhenzhe Zheng, Conghui He, Jingren Zhou, Fan Wu |
| **机构** | Tsinghua University, Alibaba DAMO Academy, Microsoft Research, ByteDance, 多个高校 |
| **领域标签** | `Data Preparation` `LLM` `Data Cleaning` `Data Integration` `Data Enrichment` `Survey` |
| **桶位** | WEAK |

---

## 方法概述

大规模数据质量是电商平台的核心痛点（商品标题噪声、属性缺失、图文不一致、类目标错等），传统规则/模板方法难以覆盖语义层面的复杂性。本 survey 系统梳理了 LLM 用于**生产就绪（application-ready）数据准备**的最新进展，标志着从规则驱动、模型特定的 pipeline 向**prompt 驱动、上下文感知、agent 化**工作流的范式转移。

覆盖三大任务类别：
1. **数据清洗（Data Cleaning）**：
   - 标准化（Standardization）：统一格式、单位、编码
   - 错误处理（Error Processing）：检测并修复拼写、逻辑、约束违反
   - 缺失值填充（Imputation）：基于上下文语义推断缺失属性

2. **数据集成（Data Integration）**：
   - 实体匹配（Entity Matching）：跨数据源同实体链接
   - Schema 匹配（Schema Matching）：跨表字段对齐
   - 知识融合：多源知识库合并

3. **数据增强（Data Enrichment）**：
   - 数据标注（Annotation）：LLM 作为弱监督标注器
   - 数据画像（Profiling）：自动分析数据集特征
   - 数据生成：合成训练数据

---

## 故事弧线 / Story Arc

**电商/内容平台大规模数据存在系统性质量问题，规则 pipeline 无法处理语义噪声** → LLM 的语义理解能力使其成为天然的数据准备工具 → survey 系统梳理 LLM 在清洗/集成/增强三大任务的应用，归纳范式转移 → 指出 hallucination、规模成本、领域知识注入三大挑战。

---

## 创新性分析

- **系统性分类框架**：对 LLM 数据准备按任务 × 技术路线做全面分类，填补 survey 空白。
- **Agentic 工作流趋势**：重点梳理 LLM 从"单次 prompt 调用"到"多步 agent 协作"的演化，对复杂数据治理 pipeline 设计有指导意义。
- **强调研究者与从业者**：Survey 定位于"application-ready"，注重方法的落地可行性评估。
- **局限**：Survey 工作本身创新性受限；没有提出新方法；部分实验数据已有论文覆盖。

---

## 关键指标

- Survey 类论文，无单一 SOTA delta；
- 覆盖 Entity Matching / Schema Matching / Imputation / Annotation 等子任务主要 SOTA 方法的性能对比（各子任务详见原文）。

---

## 评分 (满分 100)

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 14 / 30 | Survey 类，无新方法；分类框架有价值 |
| 实验指标 | 6 / 15 | Survey，无单一指标 |
| 实验质量 | 10 / 15 | 覆盖广度足，方法对比系统 |
| 效率 | 5 / 10 | 取决于具体方法 |
| 泛化性 | 5 / 5 | 三大任务横跨多领域 |
| 相关性 | 18 / 25 | 电商商品数据清洗、标注质量直接相关 |
| **Total** | **58** |

> **日期注**: 本文提交于 2026-01-22，非 2026-06-04 主窗口，为 catch-up 发现。
