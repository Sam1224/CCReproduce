# Daily AI Paper Inspection — 2026-06-08

> **Domain**: E-commerce content ecosystem & influencer (达人) governance  
> **Run time**: 2026-06-09 00:34 UTC (scheduled 08:30 GMT+8)  
> **Window**: Papers indexed on arXiv new-submission listing for **2026-06-08 (GMT+8)**

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/recent (list page) | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv search page | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/date/2026-06-08 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 (WebSearch used) |
| Google Research blog | blog | no | not attempted (rate-limit risk after DeepMind 403) | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | no | not attempted (not domain-relevant) | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (WebSearch: no new papers June 8) |
| Meituan Tech | blog | yes | 403 Forbidden | 0 (WebSearch: no new papers June 8) |
| Qwen Blog | blog | no | not attempted (low priority) | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| 机器之心 jiqizhixin.com | wechat | yes (WebSearch) | no matching June 8 articles | 0 |
| 新智元 mp.weixin.qq.com | wechat | yes (WebSearch) | no matching June 8 articles | 0 |
| OpenAI News | blog | yes (WebSearch) | no domain-relevant papers June 8 | 0 |
| DeepSeek (WebSearch) | blog | yes (WebSearch) | no new papers June 8 | 0 |
| Xiaohongshu/RedTech (WebSearch) | wechat | yes (WebSearch) | no papers found | 0 |
| Kuaishou Tech (WebSearch) | wechat | yes (WebSearch) | 1 (OneReason via arxiv) | 1 |
| Tencent Hunyuan (WebSearch) | blog | yes (WebSearch) | no papers found | 0 |
| Semantic Scholar API | aggregator | yes | 403 Forbidden | 0 |
| intoai.pub weekly digest | aggregator | yes | 403 Forbidden | 0 (WebSearch gave titles) |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| **WebSearch fallback (arxiv 2606)** | aggregator | yes | 200 OK | 4 |

**Summary**: 4/28 sources yielded candidates. arxiv direct/listing pages all returned 403 (IP-level block from sandbox). All discovery done via WebSearch fallback against `arxiv.org/abs/2606.*` identifiers. MiniMax M3 found via blog but excluded (not a paper/preprint, model release only).

---

## Picked Papers

| # | Title | arXiv | Bucket | Score |
|---|-------|-------|--------|-------|
| 1 | UNIVID: Unified Vision-Language Model for Video Moderation | [2606.05748](https://arxiv.org/abs/2606.05748) | STRONG | **87** |
| 2 | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce | [2606.05671](https://arxiv.org/abs/2606.05671) | STRONG | **77** |
| 3 | OneReason Technical Report | [2606.06260](https://arxiv.org/abs/2606.06260) | STRONG | **73** |
| 4 | Beyond Generative Decoding: Discriminative Hidden-State Readout from Omni-Modal LLM for MSA | [2606.05713](https://arxiv.org/abs/2606.05713) | WEAK | **55** |

> Papers with score ≥ 80: **UNIVID (87)** → code reproduced in `code/UNIVID/`

---

## Quick Highlights

### 🔥 Top Pick — UNIVID (87/100)
ByteDance replaces **1,000+ policy-specific models** with a single VLM that generates *policy-aware captions* as an interpretable intermediate layer. Three-stage pipeline (Risk Filter → Moderation Actor → Trend Governance) built on top of caption embeddings. Production deployment: **-42.7% violation leakage**, **-37.0% overkill rate**.

### 🛒 E-commerce Agent — QueryAgent-R1 (77/100)
Alibaba International bridges the **CTR-CVR gap** in e-commerce query recommendation. An agentic RL agent generates queries, validates them via live product retrieval, and is trained with a consistency reward that jointly optimises query click-through and downstream product conversion. Deployed at tens-of-millions DAU scale.

### 🎮 Industrial Reasoning — OneReason (73/100)
Kuaishou's OneRec team reports a **counterintuitive finding**: injecting chain-of-thought reasoning into their generative recommendation backbone shows *no advantage* over the non-thinking mode. The report diagnoses why meaningful CoT cannot be constructed from item-ID tokens and charts the path forward.

### 💬 Efficient MSA — Beyond Generative Decoding (55/100)
Universiti Sains Malaysia fits a discriminative regression head on Qwen2.5-Omni-7B's hidden state (no autoregressive decoding) for multimodal sentiment analysis, training the full 7B pipeline on a single RTX 5090 with only 1.14% trainable parameters via QLoRA.
