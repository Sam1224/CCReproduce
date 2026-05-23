# A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems

## 基本信息 / Basic Info

| 字段 | 值 |
|------|-----|
| **Title** | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems |
| **arXiv** | https://arxiv.org/abs/2605.09338 |
| **Authors** | Yiming Zhu, Xu Liu, Ziyun Xu, Zheng Wu, Joena Zhang, Sirius Chen, Chenheli Hua, Silvester Yao, Qichao Que, Wentao Shi, Junfeng Pan, Linhong Zhu |
| **Affiliation** | Meta Platforms, Menlo Park, USA |
| **Submitted** | May 10, 2026 |
| **Venue** | SIGIR '26 (ACM SIGIR Conference on Research and Development in Information Retrieval, July 2026) |
| **Domain Tags** | `multimodal-recommendation` `LLM-caption` `DLRM` `industrial-scale` `user-generated-content` |
| **Code** | Not released |
| **Bucket** | STRONG |
| **Total** | **73** |

---

## 方法概述 / Method Overview

**中文：**
该论文提出了一个工业级多模态 LLM（MM-LLM）驱动的多媒体内容理解通用框架，用于大规模推荐系统。框架采用**三段式架构（tripartite architecture）**：(1) **内容解释模块**——基于 LLaMA2 的 MM-LLM 通过 Q-Former 对齐视觉特征到语言空间，生成描述性字幕；(2) **表征抽取模块**——将字幕 token 化为与 DLRM（Deep Learning Recommendation Model）兼容的类别特征向量；(3) **流水线集成模块**——在工业延迟约束下，使用 1.5B 参数的 LLaMA2 精简版实现实时字幕生成。核心思想是用 MM-LLM 蒸馏视觉、文本和声学信号为自然语言描述，再注入已有的推荐 DLRM 栈。

**English:**
This paper proposes a general framework for MM-LLM-driven multimedia understanding in large-scale recommendation systems, validated at Meta. The tripartite architecture comprises: (1) Content interpretation via a LLaMA2-based MM-LLM with Q-Former alignment generating descriptive captions; (2) Representation extraction tokenizing captions into categorical feature vectors compatible with DLRM; (3) Pipeline integration using a compact 1.5B-parameter LLaMA2 variant for real-time caption generation under industrial latency constraints. The key insight is distilling multimodal signals (visual, text, audio) into natural language captions that are then ingested as enriched features by the existing recommendation stack.

---

## 故事线 / Story Arc

传统推荐系统对多媒体内容的高维语义信号利用不足 →  
MM-LLM 具备强大的多模态理解能力，但其直接集成受限于工业延迟约束 →  
设计三段式框架：MM-LLM 生成字幕 → 字幕 token 化为 DLRM 特征 → 1.5B 精简 LLM 满足实时要求 →  
在 Meta 平台上验证：LLM 字幕作为特征显著提升用户偏好建模保真度。

---

## 创新性分析 / Innovation

- **工业落地价值高**：明确解决了 MM-LLM 与工业推荐系统（DLRM）的接口问题，有清晰的工程路径。
- **三段式解耦**：内容理解、表征提取、管线集成三层分离，易于各层独立升级，是系统级设计贡献。
- **轻量化 LLM**：专门训练 1.5B 参数变体满足实时要求，平衡能力与延迟。
- **局限**：Q-Former + LLaMA2 的组合不是全新概念（BLIP-2 等已有先例），框架本身是已知模块的工程组合，学术创新度有限；但工业贡献价值高。

---

## 关键指标 / Key Metrics

| Setting | Validation | Notes |
|---------|-----------|-------|
| Meta 推荐系统实际部署 | A/B 测试 | 提升用户偏好建模保真度（具体数字未公开） |
| SIGIR '26 接收 | 顶级 IR 会议 | 工业轨道，Meta 内部验证 |

---

## 评分 / Scoring

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 18 / 30 | 框架级贡献，组件（Q-Former/LLaMA2/DLRM）均已有先例；三段式解耦是系统设计贡献 |
| 实验指标 | 10 / 15 | Meta 规模实际部署验证，但具体指标数字未披露 |
| 实验质量 | 11 / 15 | 工业 A/B 测试背书，SIGIR '26 接收；消融细节未披露 |
| 效率 | 7 / 10 | 1.5B 参数专项优化满足实时推理约束 |
| 泛化性 | 4 / 5 | 框架设计通用，适应图、视频、文本多种 UGC 内容类型 |
| 相关性 | 23 / 25 | 直接针对电商 / 内容平台大规模推荐中的多媒体理解，与字节、阿里等场景高度对齐 |
| **Total** | **73** | |
