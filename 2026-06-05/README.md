# Daily AI Paper Inspection — 2026-06-05 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Run time:** 2026-06-06 00:36 UTC (08:36 GMT+8)  
**Yesterday (GMT+8):** 2026-06-05  
**Target arXiv ID range:** 2606.04xxx – 2606.06xxx (submissions ~June 3-5, 2026)

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
| arxiv cs.AI/current (WebSearch fallback) | aggregator | yes | 200 (via WebSearch) | 3 |
| arxiv cs.LG/current (WebSearch fallback) | aggregator | yes | 200 (via WebSearch) | 2 |
| HuggingFace papers/date/2026-06-05 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace trending (WebSearch fallback) | aggregator | yes | 200 (via WebSearch) | 3 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog (WebSearch fallback) | blog | yes | 200 (via WebSearch) | 1 (Qwen3.7-Plus, not a paper) |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| 量子位 (WebSearch fallback) | wechat | yes | 200 (via WebSearch) | 0 |
| 机器之心 jiqizhixin.com (WebSearch) | wechat | yes | 200 (via WebSearch, no results) | 0 |
| 新智元 mp.weixin.qq.com (WebSearch) | wechat | yes | 200 (via WebSearch) | 0 (articles not papers) |
| OpenAI News (WebSearch) | blog | yes | 200 (via WebSearch) | 0 |
| DeepSeek (WebSearch) | blog | yes | 200 (via WebSearch) | 0 (V4 from April 2026) |
| Xiaohongshu REDtech (WebSearch) | wechat | yes | 200 (via WebSearch, access blocked) | 0 |
| Kuaishou Tech (WebSearch) | wechat | yes | 200 (via WebSearch) | 1 (OneReason) |
| Tencent Hunyuan (WebSearch) | blog | yes | 200 (via WebSearch) | 0 (no June 5 paper) |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| arXiv 2606 targeted WebSearch (cs.CL/IR/CV/LG/AI) | aggregator | yes | 200 (via WebSearch) | 12 candidates → 6 picked |

**Total:** 29 sources attempted (25 actually called), 6 candidate papers selected.

---

## Selected Papers

| # | Paper | arXiv ID | Affiliation | Bucket | Score |
|---|-------|----------|-------------|--------|-------|
| 1 | UNIVID: Unified Vision-Language Model for Video Moderation | 2606.05748 | ByteDance | **STRONG** | **87** |
| 2 | ANCHOR: Agentic Noise Creation Framework for Human Simulation and Denoising Recommendation | 2606.05621 | Unknown | **STRONG** | **84** |
| 3 | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation | 2606.05671 | Alibaba INTL | **STRONG** | **79** |
| 4 | OneReason Technical Report | 2606.06260 | Kuaishou | **STRONG** | **69** |
| 5 | Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems | 2606.04306 | McGill/UCLA/NYU/… | **WEAK** | **61** |
| 6 | DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities | 2606.04205 | Academia | **WEAK** | **59** |

---

## Code Reproductions (Score ≥ 80)

| Paper | Model Name | Folder |
|-------|-----------|--------|
| UNIVID | UNIVID | `code/UNIVID/` |
| ANCHOR | ANCHOR | `ANCHOR/` (pre-existing) |

---

## Paper Detail Files

- [UNIVID](papers/univid.md)
- [ANCHOR](papers/anchor.md)
- [QueryAgent-R1](papers/queryagent_r1.md)
- [OneReason](papers/onereason.md)
- [Organizational Control Layer](papers/ocl.md)
- [DetectZoo](papers/detectzoo.md)
