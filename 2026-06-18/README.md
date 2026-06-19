# Daily AI Paper Report — 2026-06-18 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Focus:** LLM / VLM / MLLM · Captioning · Embeddings · Clustering · Agents · RAG · Distillation · Violation Detection · Content Governance · Data Quality

---

## Source Coverage

> arXiv `/list/` pages all returned **HTTP 403 Forbidden** in this sandbox (network policy blocks the listing endpoint). All arXiv discovery was therefore performed via **WebSearch** fallback. Semantic Scholar API also returned 403. HuggingFace papers pages returned 403. Lab blog pages (DeepMind, Google Research, Meta) returned 403. Chinese news sites returned 403. Results below are the genuine outcomes of every attempted call.

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL /new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV /new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR /new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM /new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG /new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI /new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/date/2026-06-18 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 1 (via WebSearch: AI Control Roadmap, Jun 18) |
| Google Research blog | blog | yes | 403 Forbidden | 0 (via WebSearch) |
| Meta AI Research | blog | yes | 403 Forbidden | 0 (via WebSearch, recent papers unrelated) |
| Anthropic News | blog | yes | 403 Forbidden | 0 (via WebSearch) |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (via WebSearch, last papers from Apr 2026) |
| Meituan Tech (tech.meituan.com) | blog | yes | 403 Forbidden | 0 (via WebSearch) |
| Qwen Blog (qwenlm.github.io) | blog | yes | 403 Forbidden | 0 (via WebSearch, no Jun 18 post) |
| 量子位 QbitAI (qbitai.com) | wechat | yes | 403 Forbidden | 0 (via WebSearch) |
| 机器之心 (jiqizhixin.com) | wechat | yes (WebSearch fallback) | no direct access | 0 |
| 新智元 (mp.weixin.qq.com) | wechat | yes (WebSearch fallback) | no direct access | 0 |
| OpenAI News (openai.com) | blog | yes (WebSearch fallback) | no direct access | 0 (only recruitment news on Jun 18) |
| DeepSeek (deepseek.com) | blog | yes (WebSearch fallback) | no direct access | 0 |
| Xiaohongshu / REDtech | wechat | yes (WebSearch fallback) | no direct access | 0 |
| Kuaishou Tech | wechat | yes (WebSearch fallback) | no direct access | 0 |
| Tencent Hunyuan | wechat | yes (WebSearch fallback) | no direct access | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| arxiv cs.CL WebSearch fallback | aggregator | yes | OK | 4 |
| arxiv cs.CV WebSearch fallback | aggregator | yes | OK | 2 |
| arxiv cs.IR WebSearch fallback | aggregator | yes | OK | 3 |
| arxiv cs.LG WebSearch fallback | aggregator | yes | OK | 2 |
| arxiv cs.AI WebSearch fallback | aggregator | yes | OK | 1 |

**Note on date window:** arXiv IDs for June 2026 follow the `2606.XXXXX` format. Papers with IDs in the range `2606.15xxx–2606.19xxx` correspond to submissions made approximately June 14–18, 2026. All selected papers have `2606.` prefix confirming June 2026 submission; exact June 18 papers land around `2606.19xxx` but cannot be precisely isolated without direct listing access.

---

## Selected Papers — Scores & Picks

| # | Title | arXiv | Score | Bucket |
|---|-------|-------|-------|--------|
| 1 | QueryAgent-R1: E-Commerce Query Recommendation via RL+Retrieval | 2606.05671 | **85** | STRONG |
| 2 | Beyond Item IDs: Semantic-Native Short-Form-Video Rec at Scale | 2606.07546 | **80** | STRONG |
| 3 | Detecting AI-Generated Content on Social Media with MLLMs | 2606.11200 | **79** | STRONG |
| 4 | OneFeed: Unified Generative Framework for Feed + Query | 2606.07972 | **77** | STRONG |
| 5 | RL+CoT for Explainable Hateful & Propagandistic Meme Detection | 2606.15307 | **74** | STRONG |
| 6 | MAGE-RAG: Multigranular Adaptive Graph Evidence for Multimodal RAG | 2606.15906 | **71** | WEAK |
| 7 | Implicit Reasoning for LLM-based Generative Recommendation | 2606.14142 | **69** | WEAK |

**Code reproductions** (score ≥ 80): [`code/QueryAgent-R1/`](../code/QueryAgent-R1/) · [`code/SemanticNativeSFVRec/`](../code/SemanticNativeSFVRec/)

---

## Paper Details

- [papers/queryagent-r1.md](papers/queryagent-r1.md) — Score 85
- [papers/beyond-item-ids.md](papers/beyond-item-ids.md) — Score 80
- [papers/aigc-detection-social.md](papers/aigc-detection-social.md) — Score 79
- [papers/onefeed.md](papers/onefeed.md) — Score 77
- [papers/rl-cot-meme-detection.md](papers/rl-cot-meme-detection.md) — Score 74
- [papers/mage-rag.md](papers/mage-rag.md) — Score 71
- [papers/implicit-reasoning-llm-rec.md](papers/implicit-reasoning-llm-rec.md) — Score 69
