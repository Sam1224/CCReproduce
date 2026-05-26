# GranuRAG: From Scenes to Elements — Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG |
| **arXiv** | [2605.15019](https://arxiv.org/abs/2605.15019) |
| **Authors** | Guanhua Chen, Chuyue Huang, Yutong Yao, Shudong Liu, Xueqing Song, Lidia S. Chao, Derek F. Wong |
| **Affiliations** | (University of Macau 等) |
| **Date** | 2026-05-14 |
| **Bucket** | WEAK |
| **Total** | **57 / 100** |

---

## 故事弧 / Story Arc

> **问题:** 多模态RAG系统以粗粒度检索证据（整张图片/场景），与细粒度用户查询之间存在粒度不匹配，导致检索失败难以溯源，可验证性差。
>
> **方案:** GranuRAG框架：(1) 元素级检测与分类——将视觉场景分解为细粒度元素；(2) 多粒度跨模态对齐检索——在元素级别进行证据检索；(3) 归因约束生成——生成答案时对检索证据进行精确归因。
>
> **基准贡献:** GranuVistaVQA——基于真实地标的多粒度VQA数据集，包含元素级标注和多视角图片。

---

## 方法概述 / Method Summary

```
Query (text) + Visual Database (images)
           ↓
1. Element-Level Detection & Classification
   - 视觉场景 → 检测细粒度元素（标志、建筑细节等）
   - 元素分类并标注
           ↓
2. Multi-Granularity Cross-Modal Alignment
   - 元素级视觉特征 ↔ 文本查询对齐
   - 场景级 + 元素级双粒度检索
           ↓
3. Attribution-Constrained Generation
   - 生成答案时显式引用检索到的元素证据
   - 可追溯、可验证的推理链
```

---

## 关键指标 / Key Metrics

| Benchmark | Metric | GranuRAG | Baseline |
|-----------|--------|----------|----------|
| GranuVistaVQA | Accuracy | +↑ | Coarse-grained RAG |
| Attribution accuracy | — | 元素级归因明确 | 无法溯源 |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 18 | 30 | 元素级检索粒度改进，自然延伸；非突破性创新 |
| Experimental SOTA delta | 8 | 15 | 地标VQA任务改进；应用范围较窄 |
| Experimental quality / ablations | 9 | 15 | 新数据集贡献有价值；消融元素/场景粒度 |
| Efficiency | 5 | 10 | 细粒度检索增加检索时间 |
| Generalization | 3 | 5 | 地标场景为主，其他场景迁移待验证 |
| Domain relevance (ecom + governance) | 14 | 25 | RAG改进间接相关；可用于商品细节问答 |
| **Total** | **57** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **商品细节问答:** 电商平台商品细节查询（材质、尺寸等）可受益于细粒度RAG
- **内容审核溯源:** 可验证的归因链有助于内容违规溯源
