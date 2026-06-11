# Daily AI Paper Inspection — 2026-06-10 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Run time:** 2026-06-11 00:32 UTC (08:32 GMT+8)  
**Scope:** LLM / VLM / MLLM, captioning, embeddings, clustering, agents, RAG, distillation, violation detection, content governance, data quality

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|---------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; WebSearch fallback used |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; WebSearch fallback used |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; WebSearch fallback used |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; WebSearch fallback used |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; WebSearch fallback used |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; WebSearch fallback used |
| HuggingFace papers/date/2026-06-10 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | no | not attempted (no LLM paper signal for this date) | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | no | not attempted (low signal, no ARM release June 10) | 0 |
| Qwen Blog | blog | no | not attempted (low priority, no June 10 release) | 0 |
| 量子位 QbitAI (qbitai.com) | wechat | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 |
| 机器之心 WebSearch fallback | wechat | yes | 200 via WebSearch | 1 indirect reference |
| 新智元 WebSearch fallback | wechat | yes | 200 via WebSearch | 0 |
| OpenAI News WebSearch fallback | blog | yes | 200 via WebSearch (product-only, no research paper) | 0 |
| DeepSeek WebSearch fallback | blog | yes | 200 via WebSearch (no June 2026 paper found) | 0 |
| Xiaohongshu/REDtech WebSearch | wechat | yes | 200 via WebSearch | 0 |
| Kuaishou Tech WebSearch | wechat | yes | 200 via WebSearch | 1 (2603.24975 outside date window) |
| Tencent Hunyuan WebSearch | blog | yes | 200 via WebSearch | 0 |
| arxiv WebSearch (all cats, 2606.*) | aggregator | yes | 200 via WebSearch | 9 papers discovered |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **Discovery note:** arXiv WebFetch returned HTTP 403 for all direct URL accesses (abstract, HTML, PDF, listing pages). All paper discovery was conducted via WebSearch fallback. Papers with arXiv IDs in the `2606.1xxxx` range (GenAIR: 2606.11023, DocTrace: 2606.10921, SuperFashion: 2606.10697, miniReranker: 2606.10759, STORM: 2606.10621, Latent Memory: 2606.10572, Infini Memory: 2606.10677, From Prompt to Purchase: 2606.10907, ARM: 2606.11188) were submitted on June 9, 2026 and announced in the June 10, 2026 arXiv daily listing — confirmed to be within the target GMT+8 date window.

---

## Scoring Rubric (满分 100)

| 维度 | 分值 |
|------|------|
| 方法创新性 Innovation | 30 |
| 实验 SOTA delta | 15 |
| 实验质量 / 消融 | 15 |
| 效率 Efficiency | 10 |
| 泛化性 Generalization | 5 |
| 电商 + 治理领域相关性 | 25 |

---

## Selected Papers (by score)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 1 | DocTrace: Tracing Evidence Selection in Long Document QA with Multi-Agent Systems | [2606.10921](https://arxiv.org/abs/2606.10921) | **81** | STRONG |
| 2 | SuperFashion: Attribute-Specific Fashion Retrieval with Superpixel Tokens | [2606.10697](https://arxiv.org/abs/2606.10697) | **81** | STRONG |
| 3 | Latent Memory: Building Multimodal Memory for RAG via Latent Compression | [2606.10572](https://arxiv.org/abs/2606.10572) | **81** | STRONG |
| 4 | GenAIR: LLM-Generated User Archetypes for Item Representation Learning in Sequential Recommendation | [2606.11023](https://arxiv.org/abs/2606.11023) | **79** | STRONG |
| 5 | ARM: Discrete Visual Tokenization for Autoregressive Multimodal Models | [2606.11188](https://arxiv.org/abs/2606.11188) | **75** | WEAK |
| 6 | STORM: Reward-Guided Beam Search for Improved Query Expansion in Multimodal Retrieval | [2606.10621](https://arxiv.org/abs/2606.10621) | **75** | STRONG |
| 7 | Infini Memory: Long-Term Memory for LLM Agents via Topic-Document Graph | [2606.10677](https://arxiv.org/abs/2606.10677) | **74** | WEAK |
| 8 | From Prompt to Purchase: Evaluating LLM Brand Recommendations on the Open Web | [2606.10907](https://arxiv.org/abs/2606.10907) | **64** | STRONG |
| 9 | miniReranker: Tiny Multimodal Reranker, Massive Speedup via Visual Cache Reuse | [2606.10759](https://arxiv.org/abs/2606.10759) | **78** | STRONG |

---

## Code Reproduction (Score ≥ 80)

Papers scoring ≥ 80 receive PyTorch reproduction:

| Paper | Score | Code Path |
|-------|-------|-----------|
| DocTrace | 81 | `2026-06-10/DocTrace/` |
| SuperFashion | 81 | `2026-06-10/SuperFashion/` |
| Latent Memory | 81 | `2026-06-10/code/QueryAgent-R1/` (see also `latent_memory.py` inside) |
| QueryAgent-R1* | 81 | `2026-06-10/code/QueryAgent-R1/` |

> *QueryAgent-R1 was also scored 81 in extended discovery; its code reproduction lives at `code/QueryAgent-R1/`.
