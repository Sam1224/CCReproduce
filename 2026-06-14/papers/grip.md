## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | GRIP: Feedback-Guided Prompt Retrieval for Large Multimodal Models |
| **作者** | Garvita Allabadi, Matteo Sodano, Roberto Estevão, Yuxiong Wang, Vikram Adve, Emre Kıcıman, Ranveer Chandra |
| **机构** | University of Illinois Urbana-Champaign, University of Bonn, Microsoft |
| **arXiv** | [2606.12744](https://arxiv.org/abs/2606.12744) |
| **发布日期** | 2026-06-12 |
| **领域标签** | 多模态 VLM/MLLM、Prompt Retrieval、ICL、in-context learning、Caption |

---

## 方法概述

**Story Arc**：多模态 in-context learning 依赖检索示例，但按 embedding 相似度取示例不等于"对模型有用"——相似示例可能误导模型（如 caption 任务需要关键细节差异）。GRIP 把检索监督改成来自 LMM 的反馈：(1) 对候选邻居逐一做"加入 vs 不加入"对比实验；(2) 按任务指标增益标注为 beneficial 或 detrimental；(3) 用对比学习训练一个 **vision-only retriever**，使检索空间直接对齐"带来性能提升"的 utility。

**特点**：Retriever 用开源 LMM 反馈训练，**跨模型可迁移**到闭源模型（GPT-4o 等），降低落地成本。

**Innovation vs Prior Work**：EPR 等 feedback-guided ICL retriever 针对 NLP 任务；GRIP 首次把 utility alignment 扩展到多模态场景，并用 vision-only encoder 缩小 train-deploy gap。

---

## 关键指标

| 数据集 | 模型 | GRIP | CLIP/ViT (similarity) |
|--------|------|------|----------------------|
| Oxford Pets (Classification) | Qwen2.5-VL-7B | **83.9%** | 79.0% (ViT) |
| ScienceVQA | Qwen2.5-VL-7B | **84.2%** | 81.9% (DINO) |
| COCO CIDEr (Caption) | Qwen2.5-VL-7B | **72.5** | 72.0 (DINO) |
| ScienceVQA | GPT-4o | **85.9%** | 84.8% (CLIP) |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 21/30 | Utility-aligned retrieval for multimodal ICL 有新意；跨模型迁移有价值 |
| 实验指标 | 11/15 | 多任务提升明显，跨模型验证 |
| 实验质量 | 12/15 | 消融充分，hard negative 设计分析到位 |
| 效率 | 7/10 | Vision-only retriever 轻量，训练代价在 feedback 收集 |
| 泛化性 | 4/5 | 跨任务、跨模型均有验证 |
| 领域相关性 | 17/25 | 适用于多模态内容理解/Caption RAG，但为通用方法 |
| **Total** | **72/100** | 对多模态 RAG 示例检索有较强参考价值 |
