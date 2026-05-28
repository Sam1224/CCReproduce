# Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini |
| **Authors** | Madhuri Shanbhogue et al. (88 authors, Google DeepMind) |
| **Affiliations** | Google DeepMind |
| **arXiv** | [2605.27295](https://arxiv.org/abs/2605.27295) |
| **Submitted** | 2026-05-26 (listed 2026-05-27 GMT+8) ✓ |
| **Bucket** | STRONG |
| **Total** | **84 / 100** |
| **Code** | `code/GeminiEmbedding2/` |

---

## 方法概述 / Method Overview

### EN
Existing embedding models handle only one or two modalities (text, image), leaving a gap for unified multimodal representations needed for RAG, recommendation, and search systems. Gemini Embedding 2 (GE2) is the first *natively* multimodal embedding model that jointly embeds **video, audio, image, and text** in a single unified representation space, built on top of the Gemini foundation model. Training uses large-scale contrastive learning in a **multi-task, multi-stage** setup: an initial stage trains on broad web-scale pairs, then domain-specific stages fine-tune on retrieval, semantic similarity, clustering, and QA tasks simultaneously. A key novelty is that Gemini's native multimodal decoder architecture is repurposed as an encoder by using late-layer hidden states as embeddings. This avoids modality-specific projectors and enables cross-modal and mixed-modal queries (e.g., image+text query against a video corpus) without any task-specific heads.

### ZH
现有嵌入模型通常只处理单一或双模态（文本、图像），无法统一表示视频、音频、图像与文本。Gemini Embedding 2（GE2）是首个**原生多模态**嵌入模型，基于 Gemini 基础模型，将视频、音频、图像、文本全部映射到同一向量空间。训练采用**多任务、多阶段**大规模对比学习：首阶段在海量网络数据上训练，后续阶段同时在检索、语义相似度、聚类、问答等任务上联合微调。核心创新在于直接利用 Gemini 解码器的后层隐状态作为嵌入向量，无需模态专用投影头，支持混合模态查询（如图文混合查询视频库）。

---

## 故事弧 / Story Arc

> **"现有嵌入模型模态单一"** → 我们设计了原生多模态嵌入模型 GE2，用多任务多阶段对比学习在统一空间中表示视频/音频/图像/文本，在多个基准上达到 SOTA。

---

## 创新性分析 / Innovation Analysis

1. **原生多模态 vs. 拼接方法**：与 CLIP 系列（image+text only）或 BLIP 系列不同，GE2 支持全四模态（含视频、音频），且不依赖模态专用投影层。
2. **多任务统一训练**：一个模型覆盖检索、聚类、语义相似度、分类、RAG 等所有下游任务。
3. **解码器转编码器**：将 Gemini 解码器的隐层特征直接用于嵌入，规避了 encoder-only 模型在多模态扩展上的限制。
4. **零样本泛化**：跨天文、生物、美食、艺术等专业领域实现稳健零样本性能。

可行性：技术路线清晰，Google 有资源支撑，已在 Google Cloud Vertex AI 上线。

---

## 关键指标 / Key Metrics

| Dataset/Benchmark | Metric | GE2 | Baseline |
|-------------------|--------|-----|---------|
| MTEB Multilingual | avg | **69.9** | ~67 (prev. SOTA) |
| MTEB Code | avg | **84.0** | ~81 |
| MSCOCO (image-text retrieval) | R@1 | **62.9** | ~58 |
| Vatex (video-text retrieval) | NDCG@10 | **68.8** | ~63 |

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 25 | 首个原生四模态统一嵌入；多任务多阶段训练 |
| Experimental SOTA delta | 15 | 12 | 多个基准 SOTA，但部分 baseline 不够明确 |
| Experimental quality / ablations | 15 | 12 | 大规模多基准评测；缺少详细 ablation |
| Efficiency | 10 | 8 | 统一模型降低部署成本；推理成本高 |
| Generalization | 5 | 5 | 跨领域零样本强 |
| Domain relevance | 25 | 22 | 直接适用于电商多模态搜索、RAG、达人内容理解 |
| **Total** | **100** | **84** | |

---

## 电商治理价值 / E-commerce Governance Value

GE2 的原生多模态嵌入对电商内容生态有直接价值：
- **达人内容理解**：视频+文字+音频统一嵌入，可用于直播/短视频内容聚类和违规检测
- **商品以图搜图**：跨模态检索提升图文商品匹配精度
- **RAG 增强**：支持多模态 RAG，为电商 LLM Agent 提供更丰富上下文
- **相似度计算**：统一空间中的向量相似度可直接用于重复内容检测和内容去重
