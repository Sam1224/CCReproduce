## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Personal AI Agent for Camera Roll VQA |
| **Authors** | Thao Nguyen, Krishna Kumar Singh, Donghyun Kim, Yong Jae Lee, Yuheng Li |
| **Affiliations** | University of Wisconsin-Madison, Korea University, Adobe Research |
| **ArXiv ID** | [2606.05275](https://arxiv.org/abs/2606.05275) |
| **Submitted** | 2026-06-03 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.CV, cs.AI |
| **Project** | [thaoshibe.github.io/camroll](https://thaoshibe.github.io/camroll) |
| **Bucket** | WEAK |
| **Total** | **79 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 23 | 30 | Novel task formulation (personal visual memory VQA), hierarchical memory + minimal tool set agent, real user data collection at scale |
| Experimental SOTA delta | 11 | 15 | Recall=88.5%, Acc=70.5% (MC); only ~3.2K tokens vs 750K for naive baseline |
| Experimental quality / ablations | 12 | 15 | 50 users, 31,476 images, 2,500 QA; multiple agent configurations compared; efficiency metrics explicit |
| Efficiency | 10 | 10 | Token cost reduction is ~234x vs naive baseline — primary selling point |
| Generalization | 3 | 5 | Specific to personal photo management; hierarchical retrieval architecture is transferable |
| Domain relevance | 20 | 25 | Highly relevant to multimodal content understanding and agent retrieval; less direct to e-commerce governance, but demonstrates key VLM agent patterns |
| **Total** | **79** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
用户的个人相册（camera roll）是一类独特的"长时序个人视觉记忆"：跨越多年、内容高度私密、缺乏结构化标注、数量达数万张。现有 VQA 方法要么需要将所有图片全部输入（token 成本极高），要么依赖一次性 RAG（召回精度不足）。没有专门针对个人相册的数据集和 agent 评测框架。

**X is insufficient → we design Y to solve it：**
> 通用多模态 RAG/memory 方法在个人相册场景下面临两个核心挑战：(1) 照片数量太多，全部输入不可行（750K tokens）；(2) 个人照片的视觉细节、时间连贯性和用户特定上下文（如"我常去的那家餐厅"）无法用通用 embedding 精确检索。camroll-agent 设计了层级记忆（hierarchical memory）+ 最小工具集，通过迭代式精确检索将 token 成本压缩至约 3.2K，同时保持高召回。

### 核心方法

1. **camroll 数据集**：收集自 50 名真实用户，包含 31,476 张照片和 2,500 个 QA 对，覆盖简单事实问题（"昨天吃的什么"）和开放式问题（"推荐我没吃过的菜"）。

2. **camroll-agent**：
   - **层级记忆（Hierarchical Memory）**：事件级摘要 + 图片级 metadata 双层记忆，支持粗粒度快速定位和细粒度精确检索
   - **最小工具集**：仅提供 2-3 个核心工具（时间过滤、相似度检索、详细查看），避免工具滥用
   - **迭代式导航**：agent 按需逐步检索，而非一次性加载所有图片

3. **评测指标**：Recall（图片检索命中率）、Acc（多选题准确率）、Judge（开放题 GPT-4 评分）、Token（总 token 消耗）。

### English Summary

The paper introduces camroll, a personal camera-roll VQA dataset (50 users, 31,476 images, 2,500 QA pairs) and camroll-agent, a conversational agent with hierarchical memory and a minimal tool set for navigating large personal visual memory collections. The agent iteratively retrieves relevant photos rather than loading all images upfront, reducing token cost from ~750K (naive all-images baseline) to ~3.2K while maintaining high recall (88.5%) and accuracy (70.5% on multiple-choice). The work reveals that personalized visual memory requires fundamentally different handling from standard long-context text memory, due to visual detail, temporal coherence, and user-specific semantic context.

---

## 创新点分析 / Innovation Analysis

**中文：** 将"个人视觉记忆"作为独立研究对象，区别于通用长文本记忆和通用图片检索，是本文的主要学术贡献。层级记忆（事件级摘要+图片级 metadata）的设计对电商内容场景（如：达人历史创作内容检索、用户历史购买行为理解）具有直接的方法迁移价值。token 效率的量化（234x 压缩）为实际部署提供了明确的工程参考。

**English:** Formalizing personal visual memory as a distinct research regime — separate from general long-context text or general image retrieval — is the main scholarly contribution. The hierarchical memory design (event-level summary + image-level metadata) has direct transfer value for e-commerce scenarios (e.g., influencer historical content retrieval, user visual purchase history understanding). The explicit token efficiency quantification provides a concrete engineering reference for deployment.

---

## 关键指标 / Key Metrics

| Metric | camroll-agent | All-images baseline |
|--------|--------------|---------------------|
| MC Recall | **88.5** | ~100 (trivially) |
| MC Acc | **70.5** | — |
| Free-form Recall | **83.1** | — |
| Free-form Judge | **4.11** / 5 | — |
| Total Tokens | **~3,200** | **~750,000** |
| Token reduction | **~234×** | — |
