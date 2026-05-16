# From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG |
| **arXiv ID** | 2605.15019 |
| **Submitted** | 2026-05-14 |
| **Link** | https://arxiv.org/abs/2605.15019 |
| **Authors** | Guanhua Chen, Chuyue Huang, Yutong Yao, Shudong Liu, Xueqing Song, Lidia S. Chao, Derek F. Wong |
| **Affiliation** | University of Macau |
| **Code** | Not yet public |
| **Venue** | arXiv preprint |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**问题（Story Arc）：** 现有多模态 RAG 系统在场景（scene）级别检索证据（完整图片），导致细粒度查询的证据粒度失配（一张图片可能只包含目标实体的部分视角）以及失败原因不可解释。

**解决方案：**  
- **GranuVistaVQA 基准**：针对真实地标的多粒度元素级标注基准，包含跨视角的局部观察挑战（个别图片仅覆盖部分实体），是首个强调可验证性的多模态 VQA 数据集。
- **GranuRAG**：将视觉元素（Visual Elements）作为一等检索单元的多粒度框架，包含三个阶段：
  1. **元素级检测与分类**（VLM驱动的 object detection + classification）；  
  2. **元素级检索**（基于 CLIP 的 element embedding 相似度）；  
  3. **聚合推理**（多粒度证据融合 + 答案生成）。
- 通过在元素级别而非隐式注意力层面做显式 grounding，系统可输出可验证的证据链，支持错误诊断。

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| vs. 标准图像级 RAG | 把检索粒度从整幅图下推到元素级，避免 scene-query 粒度失配 |
| vs. RAVEN / MIND-RAG | 引入可解释证据链（哪个元素支持哪个答案），而非黑箱 attention |
| 新数据集 | GranuVistaVQA 填补了元素级多视角 VQA 基准的空白 |
| 可行性 | 依赖现成 VLM 检测器和 CLIP，部署门槛低 |

---

## 关键指标 / Key Metrics

| Dataset | Metric | GranuRAG | Best Baseline |
|---------|--------|----------|---------------|
| GranuVistaVQA | Accuracy | +29.2% (relative over best of 6 baselines) | Scene-level RAG |
| GranuVistaVQA | F1 | significant improvement | GPT-4V direct |

---

## 评分明细 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 22 | 元素级检索单元 + 可验证证据链是较有价值的创新 |
| SOTA Delta | 15 | 13 | 29.2% 相对提升，远超 6 个 strong baselines |
| Experimental Quality | 15 | 11 | 新基准 + 细致 ablation，但应用场景略窄 |
| Efficiency | 10 | 5 | 三阶段流水线带来额外计算 |
| Generalization | 5 | 3 | 目前聚焦地标 VQA，迁移电商场景需验证 |
| Domain Relevance | 25 | 18 | 多模态 RAG 可用于电商商品 QA 与内容核查，间接相关 |
| **Total** | **100** | **72** | STRONG |

---

## 故事弧 / Story Arc

> "多模态 RAG 在场景级粒度检索导致细粒度查询匹配失败且不可解释 → GranuRAG 将视觉元素设为一等检索单元，配合 GranuVistaVQA 基准，实现可验证的多粒度证据检索，相比 6 个强基线提升 29.2%。"

---

## 电商/治理迁移价值

- **商品内容核查**：将商品图片解析为属性元素（颜色、形状、材质），在细粒度问答和质检中提升可解释性
- **AIGC 内容验证**：对 AI 生成商品描述进行图-文元素级对齐核查
- **直播内容理解**：基于视频帧的元素级证据检索用于商品合规验证
