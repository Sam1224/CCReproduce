# MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA |
| **Authors** | Yilong Zuo, Xunkai Li, Jing Yuan, Qiangqiang Dai, Hongchao Qin, Ronghua Li |
| **Affiliation** | (Chinese research institutions) |
| **arXiv ID** | [2606.15906](https://arxiv.org/abs/2606.15906) |
| **Submitted** | June 14, 2026 |
| **Venue** | Preprint |

---

## 方法概述 / Method Summary

### Story Arc

> **现有方法的问题**：长文档多模态问答需要系统在长PDF中定位稀疏证据，并整合文字、表格、图像、图表和复杂布局中的线索。现有RAG方法大多依赖固定Top-k文本块检索（丢失视觉和布局信息）或页面级视觉检索（保留原始页面但引入大量无关内容），形成静态权衡。
>
> **解决方案**：MAGE-RAG以页面检索为入口构建查询时证据图（evidence graph），离线构建包含页面节点和元素节点的图，编码包含关系、阅读顺序、布局邻接、节层次和语义邻居关系，在线时自适应选择最优粒度的证据子图送入阅读器。

### Technical Approach (EN)

MAGE-RAG introduces a multigranular adaptive evidence graph for long-document multimodal RAG:

1. **Offline Evidence Graph Construction**: Builds a graph with page nodes and element nodes encoding: containment, reading order, layout adjacency, section hierarchy, and semantic-neighbor relations — captures the full structural and semantic context of PDF documents.
2. **Query-time Adaptive Evidence Selection**: At inference, uses an agentic mechanism to select evidence at the optimal granularity (page-level, element-level, or sub-element-level) based on the query's evidence requirements.
3. **Multimodal Integration**: Handles text, tables, images, charts, and complex layouts within a unified graph structure.

### 创新亮点 (ZH)

- **多粒度自适应**：根据查询需求自适应选择最优证据粒度，克服固定粒度方法的静态权衡。
- **结构化证据图**：将文档的布局关系、层次关系和语义关系统一编码进图结构，比纯文本块更丰富。
- **智能体化RAG**：agentic机制使系统能根据查询复杂度动态调整检索策略。

---

## 关键指标 / Key Metrics

| Benchmark | Setting | MAGE-RAG | Prior SOTA |
|-----------|---------|----------|------------|
| Long-document multimodal QA | Accuracy | Improved | Fixed Top-k methods |
| Evidence coverage | Recall | Higher | Text-only RAG |
| Inference cost | Tokens per query | Adaptive (lower avg) | Page-level RAG |

---

## 评分详情 / Scoring Breakdown

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation | 22/30 | Novel multigranular evidence graph for multimodal RAG; adaptive evidence selection |
| Experimental SOTA delta | 11/15 | Improvements on long-document multimodal QA benchmarks |
| Experimental quality / ablations | 11/15 | Ablations on graph construction components |
| Efficiency | 7/10 | Adaptive granularity reduces average token cost |
| Generalization | 3/5 | Primarily document QA; less direct to e-commerce |
| Domain relevance (ecom+governance) | 17/25 | Multimodal RAG relevant to e-commerce product catalog QA and policy document retrieval |
| **Total** | **71/100** | |
