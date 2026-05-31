# Adapting Vision-Language Models for E-commerce Understanding at Scale

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Adapting Vision-Language Models for E-commerce Understanding at Scale |
| **Authors** | Matteo Nulli, Vladimir Orshulevich, Tala Bazazo, Christian Herold, Michael Kozielski, Marcin Mazur, Szymon Tuzel, Cees G. M. Snoek, Seyyed Hadi Hashemi, Omar Javed, Yannick Versley, Shahram Khadivi |
| **Affiliation** | (Major e-commerce/tech company — 12 authors) |
| **arXiv** | [2602.11733](https://arxiv.org/abs/2602.11733) |
| **Submitted** | 2026-02-12 |
| **Venue** | arXiv preprint |
| **Domain** | VLM Adaptation · E-Commerce · Product Understanding · Multi-Image · Attribute Extraction |
| **Bucket** | STRONG |

---

## 得分 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 18 | 30 | Systematic VLM adaptation study rather than novel architecture; key contribution is the *documented strategy* which was previously undocumented |
| SOTA delta | 10 | 15 | Substantially improves e-commerce task performance while preserving general VLM capabilities |
| Exp quality / ablations | 11 | 15 | Large-scale experimental study; comprehensive evaluation suite on real production tasks |
| Efficiency | 7 | 10 | Targeted adaptation (not full retraining); preserves efficiency of base VLM |
| Generalization | 4 | 5 | Strategy is broadly applicable; shown across multiple product categories and tasks |
| Domain relevance | 21 | 25 | Directly addresses VLM adaptation for e-commerce product understanding |
| **Total** | **71** | **100** | Solid practical contribution; valuable recipe for e-commerce VLM deployment |

---

## 方法概述 / Method Summary

### Story Arc
General-purpose VLMs (e.g., LLaVA, InternVL, Qwen-VL) achieve strong zero-shot performance on standard vision-language tasks but struggle with the **unique characteristics of e-commerce data**: attribute-centric reasoning (exactly what material is this?), multi-image product views, noisy user-generated images, and strict instruction following for structured attribute extraction. There is no published, well-tested strategy for adapting these models without sacrificing their broad capabilities.

**X is insufficient → we design Y:** Off-the-shelf VLMs underperform on e-commerce-specific tasks (attribute extraction, instruction following for structured output, multi-image product understanding) without a clear adaptation strategy → this paper provides the first documented, systematic VLM adaptation recipe for e-commerce at scale.

### Adaptation Framework

```
General-Purpose VLM (LLaVA / InternVL / etc.)
         ↓
[E-Commerce Fine-Tuning Strategy]
  1. Data curation:
     - Multi-image product sequences (multiple angles of same product)
     - Noisy image handling (low-quality user uploads)
     - Attribute-centric instruction pairs (structured Q&A format)
  2. Training objectives:
     - Instruction following for structured attribute extraction
     - Multi-image aggregation for product understanding
     - Balanced general + e-commerce task mixture (preserves broad capabilities)
         ↓
[Evaluation Suite]
  Task 1: Deep product understanding (attribute reasoning, material, style)
  Task 2: Strict instruction following (JSON format output, multi-attribute extraction)
  Task 3: Dynamic attribute extraction (open-world product categories)
  Task 4: General VLM benchmarks (verify no regression)
         ↓
E-Commerce-Adapted VLM
```

### Key Technical Findings
1. **Multi-image aggregation is critical:** Products viewed from multiple angles require cross-image reasoning that standard VLMs lack.
2. **Instruction following requires explicit tuning:** E-commerce requires highly structured outputs (JSON, attribute: value format); standard VLMs fail to maintain this consistently.
3. **General capability preservation:** A balanced fine-tuning mixture prevents catastrophic forgetting of broad VLM capabilities.
4. **Attribute-centric evaluation:** Standard VLM benchmarks are insufficient; the new evaluation suite based on real production tasks is essential for e-commerce adaptation.

---

## 核心指标 / Key Metrics

| Task | Adapted VLM | Base VLM | Improvement |
|------|-------------|----------|-------------|
| Product attribute extraction | High | Moderate | Substantial |
| Strict instruction following | High | Low-Moderate | Large |
| Dynamic attribute extraction | High | Low | Large |
| General VLM benchmarks | Maintained | Baseline | No regression |

---

## 相关性评估 / Domain Relevance

命中核心业务：
- **商品理解**：直接对应商品属性提取、图文匹配、商品质量评估等场景
- **多图商品**：电商商品通常有多张图片，多图聚合理解是关键能力
- **VLM 落地**：提供了 VLM 适配电商的系统性方法，工程价值高
- **内容标注**：适配后的 VLM 可用于大规模商品内容标注和属性抽取
- **内容治理**：属性理解能力可迁移到违规商品检测（如虚假宣传检测）
