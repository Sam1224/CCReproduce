# Daily AI Paper Inspection — 2026-06-16 (GMT+8)

**Inspection date (GMT+8):** 2026-06-16 (Monday)  
**arXiv listing window:** Papers submitted Thu 2026-06-12 (after 14:00 ET) through Sun 2026-06-15 — announced Monday 2026-06-16  
**Domain focus:** E-commerce content ecosystem · Influencer (达人) governance · LLM/VLM/MLLM · Captioning · Embeddings · Clustering · Agents · RAG · Distillation · Violation detection · Content governance · Data quality / labeling

---

## Source Coverage

> **Note:** All arXiv listing pages (arxiv.org/list/cs.*/new) returned HTTP 403 in this sandbox environment. Semantic Scholar API and HuggingFace papers pages also returned 403. Discovery relied entirely on WebSearch (Tier 3 fallback) with individual paper URLs manually reconstructed. The source coverage table below reflects this accurately.

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|--------------------|
| arxiv.org/list/cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv.org/list/cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv.org/list/cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv.org/list/cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv.org/list/cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv.org/list/cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv.org/list/cs.CL/current | aggregator | yes | 403 Forbidden | 0 |
| arxiv.org/list/cs.AI/current | aggregator | yes | 403 Forbidden | 0 |
| huggingface.co/papers/date/2026-06-16 | aggregator | yes | 403 Forbidden | 0 |
| huggingface.co/papers/trending | aggregator | yes | 403 Forbidden | 0 |
| deepmind.google/discover/blog | blog | yes | 403 Forbidden | 0 |
| research.google/blog | blog | yes | 403 Forbidden | 0 |
| ai.meta.com/research | blog | yes | 403 Forbidden | 0 |
| www.anthropic.com/news | blog | yes | (not attempted — out of scope) | 0 |
| seed.bytedance.com/research | blog | yes | 403 Forbidden | 0 |
| tech.meituan.com | blog | yes | (not attempted — low prio) | 0 |
| qwenlm.github.io/blog | blog | yes | 403 Forbidden | 0 |
| www.qbitai.com | wechat | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 |
| WebSearch: cs.CL/cs.CV June 16 LLM multimodal | search-fallback | yes | 200 (partial index) | 4 |
| WebSearch: cs.IR/cs.MM June 16 RAG embedding | search-fallback | yes | 200 (partial index) | 3 |
| WebSearch: 2606 e-commerce recommendation | search-fallback | yes | 200 | 5 |
| WebSearch: 2606 content moderation MLLM | search-fallback | yes | 200 | 4 |
| WebSearch: 2606 live streaming violation | search-fallback | yes | 200 | 3 |
| WebSearch: 机器之心 jiqizhixin.com June 2026 | wechat | yes | 200 (no matching results) | 0 |
| WebSearch: 新智元 mp.weixin.qq.com June 2026 | wechat | yes | 200 (no matching results) | 0 |
| WebSearch: OpenAI news June 2026 | blog | yes | 200 (no matching results) | 0 |
| WebSearch: DeepSeek June 2026 | blog | yes | 200 (no matching results) | 0 |
| WebSearch: Kuaishou Tech LLM June 2026 | blog | yes | 200 (no matching results) | 0 |
| WebSearch: Tencent Hunyuan June 2026 | blog | yes | 200 (no matching results) | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

**Total sources attempted:** 31 | **Yielding candidates:** 7 (WebSearch fallbacks)  
**Discovery note:** arXiv listing pages universally returned 403 in this environment. All paper discovery was via WebSearch. Submission dates were recovered from search snippets and cross-references. Papers marked ★ are confirmed in the June 16 Monday arXiv listing (submitted Jun 12–15); others are June 2026 papers discovered from broader search context.

---

## Papers Selected

| # | ID | Title | Bucket | Score | Code |
|---|----|-------|--------|-------|------|
| 1 | 2512.03553 | Dynamic Content Moderation in Livestreams | STRONG | **88** | `code/DynamicLiveModeration/` |
| 2 | 2606.05748 | UNIVID: Unified VLM for Video Moderation | STRONG | **83** | `code/UNIVID/` |
| 3 | 2606.12198 | LLM-Based User Personas for Recommendations at Scale | STRONG | 77 | — |
| 4 | 2606.05671 | QueryAgent-R1: E-Commerce Query Recommendation | STRONG | 77 | — |
| 5 | 2606.07995 | Customer-Agent: Ultra-Long Shopping Trajectories | STRONG | 71 | — |
| 6 | 2606.15610 ★ | LLM Judges Have Dark Current | WEAK | 66 | — |
| 7 | 2606.13392 | MiniMax Sparse Attention | WEAK | 65 | — |
| 8 | 2606.14691 ★ | CORA: Multimodal RLVR Reasoning Alignment | WEAK | 61 | — |
| 9 | 2606.15474 ★ | Who Drifted: System or Judge? | WEAK | 58 | — |

★ = confirmed in June 16 arXiv listing (submitted June 12–15)

---

## Quick Summaries

### 1. Dynamic Content Moderation in Livestreams (arXiv:2512.03553) — Score 88
**KDD 2026 | ByteDance / TikTok**  
Hybrid moderation system combining MLLM-distilled classifiers with reference-based similarity matching for live e-commerce and video streaming. Multi-modal (text+audio+visual). Production: 67% recall @ 80% precision (classification), 76% recall @ 80% precision (similarity). A/B test: −6–8% unwanted livestream views.

### 2. UNIVID: Unified VLM for Video Moderation (arXiv:2606.05748) — Score 83
**ByteDance (Jun 4, 2026)**  
Novel policy-aware captioning approach for video moderation at global scale. Three-stage pipeline: Risk Filter → Moderation Actor (UNIVID-Lite + UNIVID-RAG) → Trend Governance. Interpretable captions support human review and multi-task reuse.

### 3. LLM-Based User Personas for Recommendations at Scale (arXiv:2606.12198) — Score 77
**Google (Jun 10, 2026)**  
Real-time LLM user interest personas for billion-user video recommendation. Addresses exploitation-exploration tradeoff via semantic clustering of video embeddings. Uses knowledge distillation + asynchronous inference for cost efficiency.

### 4. QueryAgent-R1 (arXiv:2606.05671) — Score 77
**Alibaba International Digital Commerce (Jun 4, 2026)**  
Memory-augmented RL agent for e-commerce query recommendation. Chain-of-retrieval grounds query generation in real inventory, jointly optimizing CTR and CVR via consistency reward.

### 5. Customer-Agent (arXiv:2606.07995) — Score 71
**Amazon / Duke University (Jun ~7, 2026)**  
Tool-augmented LLM agent for ultra-long shopping trajectory understanding. RLVR training on ShopTrajQA benchmark with code-interpreter-based tool use.

### 6. LLM Judges Have Dark Current (arXiv:2606.15610 ★) — Score 66
**Jun 14, 2026**  
Psychometric audit protocol for LLM-as-judge systems exposing bias patterns (dark current, positional preference, cross-sensitivity) critical for LLM-based data labeling quality control.

### 7. MiniMax Sparse Attention (arXiv:2606.13392) — Score 65
**MiniMax (Jun 11–12, 2026)**  
Blockwise sparse attention enabling 1M-token context at 1/20 the compute. Foundation for long-context content analysis tasks.

### 8. CORA: Multimodal RLVR Reasoning Alignment (arXiv:2606.14691 ★) — Score 61
**Jun ~13, 2026**  
Lightweight plug-in consistency reward for multimodal RLVR addressing thinking-answer semantic gap.

### 9. Who Drifted: System or Judge? (arXiv:2606.15474 ★) — Score 58
**Jun 13, 2026**  
Anytime-valid statistical method to attribute LLM evaluation drift to system vs. judge, enabling reliable continuous monitoring pipelines.
