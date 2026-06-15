## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering for the Grand Egyptian Museum |
| **作者** | Rawan Hesham, Ali Ashraf, Amr Ahmed, Malak Alaa, Omar Ahmed, Omar Wagih |
| **机构** | Capital University (Faculty of Computers and AI), Egypt |
| **arXiv** | [2606.13267](https://arxiv.org/abs/2606.13267) |
| **发布日期** | 2026-06-12 |
| **领域标签** | 多模态、数据质量、RAG、端侧部署、双语、标注清洗 |

---

## 方法概述

**Story Arc**：博物馆导览应用需同时满足低延迟、离线可用、可靠无幻觉。TimeLens 把任务分为两条链：(1) **端侧识别**：YOLOv8n（5.97MB, TFLite）本地实时识别展品，先用 YOLO-World 自动标注再用空间清洗规则修 label，最后人工闭环；(2) **RAG 问答**：把用户追问通过 ChromaDB + 量化 Gemma 4 锚定到小规模知识库，双语（英/阿）支持。

核心经验：**label 质量比模型结构更决定成败**——展品细粒度类别下，数据治理优先于模型规模。

**Innovation**：端侧识别+后端 RAG 的分层架构，加上以 foundation-model auto-annotation + 规则清洗 + 人工闭环为核心的数据迭代方法学。

---

## 关键指标

| 指标 | 值 |
|------|-----|
| 识别 mAP@0.5 (51 类展品) | **0.995** |
| 识别 mAP@0.5:0.95 | **0.924** |
| 模型大小（YOLOv8n, TFLite） | **5.97 MB** |
| 端到端延迟（优化后） | **约 10s** |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 18/30 | 分层架构+数据清洗方法学有参考价值，技术集成为主 |
| 实验指标 | 10/15 | mAP 指标优秀，延迟有参考意义 |
| 实验质量 | 10/15 | 数据清洗消融有价值，但测试规模较小 |
| 效率 | 9/10 | 端侧 5.97MB 极轻量 |
| 泛化性 | 2/5 | 博物馆专项，电商场景需较大 adaptation |
| 领域相关性 | 16/25 | 数据清洗+RAG 方法论有迁移价值 |
| **Total** | **65/100** | 偏工程应用，数据治理经验可参考 |
