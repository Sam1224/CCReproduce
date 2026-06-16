# Daily AI Paper Inspection — 2026-06-15

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Focus:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality · data labeling

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|---------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; ~12 via WebSearch fallback |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; ~8 via WebSearch fallback |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; ~5 via WebSearch fallback |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; ~3 via WebSearch fallback |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; ~4 via WebSearch fallback |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden (WebFetch blocked) | 0 direct; ~3 via WebSearch fallback |
| HuggingFace papers/date/2026-06-15 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden; WebSearch fallback attempted | 1 (Seed1.5-VL referenced) |
| Meituan Tech | blog | no | reason=JS-rendered, no direct feed; WebSearch fallback limited | 0 |
| Qwen Blog | blog | no | reason=low-priority; WebSearch fallback: Qwen3 info found | 0 direct |
| 量子位 QbitAI | wechat | yes | 403 Forbidden (WebFetch); WebSearch attempted | 0 direct; 2 articles found |
| 机器之心 jiqizhixin.com (WebSearch) | wechat | yes | WebSearch attempted; no June 15 results | 0 |
| 新智元 mp.weixin.qq.com (WebSearch) | wechat | yes | WebSearch attempted; no relevant results | 0 |
| OpenAI News (WebSearch) | blog | yes | WebSearch attempted; no new June 15 paper | 0 |
| DeepSeek (WebSearch) | blog | yes | WebSearch attempted; no new June 15 paper | 0 |
| Xiaohongshu/RedTech (WebSearch) | wechat | yes | WebSearch attempted; no results | 0 |
| Kuaishou Tech (WebSearch) | blog | yes | WebSearch found 2606.10357 (AIR) | 1 |
| Tencent Hunyuan (WebSearch) | blog | yes | WebSearch attempted; no new June 15 paper | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | reason=not yet active (venue not open for listing) | 0 |
| arXiv WebSearch fallback (multi-category) | aggregator | yes | OK; papers in 2606.05XXX–2606.14XXX range | 10 candidates |

**Notes:**  
- arxiv.org blocked all direct WebFetch calls (403 for /list/, /abs/, /html/, /pdf/ paths). All arXiv papers discovered via WebSearch with targeted queries using arXiv IDs in the 2606.XXXXX range.  
- June 15 (GMT+8) announcement window = papers submitted June 11–12 in ET; arxiv IDs ~2606.13XXX–2606.14XXX confirmed. Additional strong-domain papers from June 3–10 also included due to high relevance.  
- Total: 10 candidates found; 10 scored; **3 papers with score ≥ 80 added as new discoveries** on top of 7 existing in repo.

---

## Papers This Run (All Scores)

| # | arxiv ID | Title | Affiliation | Score | Bucket |
|---|----------|-------|-------------|-------|--------|
| 1 | [2606.13533](https://arxiv.org/abs/2606.13533) | OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model | Kuaishou | **87** | STRONG |
| 2 | [2606.10357](https://arxiv.org/abs/2606.10357) | Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations | Kuaishou | **87** | STRONG |
| 3 | [2606.05671](https://arxiv.org/abs/2606.05671) | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation | Alibaba International | **86** | STRONG |
| 4 | [2606.12608](https://arxiv.org/abs/2606.12608) | Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants | Amazon | **82** | STRONG |
| 5 | [2606.09393](https://arxiv.org/abs/2606.09393) | CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning | InternLM/Shanghai AI Lab | **82** | STRONG |
| 6 | [2606.13001](https://arxiv.org/abs/2606.13001) | CFALR: Collaborative Filtering-Augmented LLM for Personalized Fashion Outfit Recommendation | PolyU / SMU / NUS | **77** | STRONG |
| 7 | [2606.14703](https://arxiv.org/abs/2606.14703) | Gaze Heads: How VLMs Look at What They Describe | MIT (Bau Lab) | **71** | WEAK |
| 8 | [2606.07688](https://arxiv.org/abs/2606.07688) | TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation | — | **70** | STRONG |
| 9 | [2606.11023](https://arxiv.org/abs/2606.11023) | Generative Archetype-Grounded Item Representations for Sequential Recommendation | — | **69** | STRONG |
| 10 | [2606.14046](https://arxiv.org/abs/2606.14046) | When Recommendation Denoising Meets Popularity Bias: Understanding and Mitigating Their Interaction | — | **56** | WEAK |

---

## Code Reproductions (score ≥ 80)

| Paper | Score | Path |
|-------|-------|------|
| OneRetrieval | 87 | `2026-06-15/OneRetrieval/` |
| Atomic Intent Reasoning (AIR) | 87 | `2026-06-15/AIR/` |
| QueryAgent-R1 | 86 | `2026-06-15/QueryAgent-R1/` |
| Shopping Reasoning Bench | 82 | evaluation benchmark, no model code |
| CapRL++ | 82 | `2026-06-15/CapRL/` |
