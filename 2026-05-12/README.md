# Daily AI Paper Inspection — 2026-05-12 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Run time:** 2026-05-13 00:39 UTC (08:39 GMT+8)  
**Window:** arXiv listing for 2026-05-12 GMT+8 (papers submitted ~May 8–11, 2026; appeared in arXiv listing ~May 11, 2026 UTC)

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv.org/list/cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| arxiv.org/list/cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| arxiv.org/list/cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| arxiv.org/list/cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| arxiv.org/list/cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| arxiv.org/list/cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| huggingface.co/papers/date/2026-05-12 | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| huggingface.co/papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback: WebSearch) |
| deepmind.google/discover/blog/ | blog | yes | 403 Forbidden | 0 |
| research.google/blog/ | blog | yes | 403 Forbidden | 0 |
| ai.meta.com/research/ | blog | yes | 403 Forbidden | 0 |
| anthropic.com/news | blog | yes | 403 Forbidden | 0 |
| seed.bytedance.com/research | blog | yes | 403 Forbidden | 0 |
| tech.meituan.com/ | blog | yes | 403 Forbidden | 0 |
| qwenlm.github.io/blog/ (low prio) | blog | yes | 403 Forbidden | 0 |
| qbitai.com/ (量子位) | wechat | yes | 403 Forbidden (WebSearch fallback) | 0 candidate papers |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| zezhishao/DailyArXiv Issue #579 (GitHub) | aggregator | yes | 200 OK | 5 candidates |
| WebSearch arXiv 2605.07xx–2605.10xx domain sweep | aggregator | yes | 200 OK | 6 candidates |
| jiqizhixin.com (WebSearch Tier 3) | wechat | yes | 200 OK | 0 in-scope papers |
| mp.weixin.qq.com "机器之心" (WebSearch Tier 3) | wechat | yes | 200 OK | 0 in-scope papers |
| mp.weixin.qq.com "新智元" (WebSearch Tier 3) | wechat | yes | 200 OK | 0 in-scope papers |
| openai.com/news (WebSearch Tier 3) | blog | yes | 200 OK | 0 research papers |
| deepseek.com (WebSearch Tier 3) | blog | yes | 200 OK | 0 new May-12 papers |
| Xiaohongshu/RedTech (WebSearch Tier 3) | blog | yes | 200 OK | 0 in-scope papers |
| Kuaishou Tech (WebSearch Tier 3) | blog | yes | 200 OK | 0 new May-12 papers |
| Tencent Hunyuan (WebSearch Tier 3) | blog | yes | 200 OK | 0 new May-12 papers |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **Discovery note:** All direct WebFetch calls to arxiv.org, huggingface.co, lab blogs, and qbitai.com returned 403 Forbidden in this sandbox. Discovery was completed via (a) WebSearch queries targeting arXiv ID ranges 2605.07xxx–2605.10xxx, (b) the zezhishao/DailyArXiv GitHub Issue #579 (titled "Latest 15 Papers – May 12, 2026"), and (c) targeted domain-keyword searches. All 6 paper entries were independently verified through multiple search result confirmations of title + abstract + authors.

---

## Papers Picked

This run adds **6 new papers** to the existing `DataMaster` entry from this date.

| # | Title | arXiv ID | Bucket | Score |
|---|-------|----------|--------|-------|
| 1 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | [2605.07982](https://arxiv.org/abs/2605.07982) | STRONG | **86** ⭐ |
| 2 | Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation | [2605.07613](https://arxiv.org/abs/2605.07613) | STRONG | **76** |
| 3 | RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in VL Content Moderation | [2605.07760](https://arxiv.org/abs/2605.07760) | STRONG | **74** |
| 4 | jina-embeddings-v5-omni: Multimodal Embeddings via Frozen-Tower Composition | [2605.08384](https://arxiv.org/abs/2605.08384) | WEAK | **69** |
| 5 | DRIP-R: Benchmark for Decision-Making under Real-World Policy Ambiguity in Retail | [2605.07699](https://arxiv.org/abs/2605.07699) | STRONG | **68** |
| 6 | Video Understanding Reward Modeling: Robust Benchmark and Performant Reward Models | [2605.07872](https://arxiv.org/abs/2605.07872) | WEAK | **67** |
| — | DataMaster: Towards Autonomous Data Engineering for ML (prior entry) | [2605.10906](https://arxiv.org/abs/2605.10906) | STRONG | see prior README |

### Code Reproductions (score ≥ 80)
- [`code/GLiGuard/`](code/GLiGuard/) — GLiGuard (score 86)

---

## Paper Details

### 1. GLiGuard (score: 86) ⭐ — STRONG · code at `code/GLiGuard/`

**Story arc:** Autoregressive guard models (7B–27B params) treat safety classification as sequential text generation, causing high latency (up to 90× slower) and poor scalability → GLiGuard adapts a 0.3B bidirectional GLiNER2 encoder to encode safety schemas as structured token sequences for single-pass, non-autoregressive classification across 14 harm categories + 11 jailbreak strategies simultaneously.

**Key metrics:** 9 safety benchmarks · F1 within 1.7 pts of 7B–27B baselines · 16× higher throughput · 17× lower latency · 23–90× smaller

→ Full details: [papers/gligard.md](papers/gligard.md)

---

### 2. Intent-Driven SID Recommendation (score: 76) — STRONG

**Story arc:** Implicit user intents in conversational news recommendation bypass standard RAG keyword retrieval, causing a fundamental retrieve-first bottleneck → intent-driven Semantic ID (SID) generation under Generate-then-Match maps 6 intent types to hierarchical SID prefixes via GPT-4 CoT distillation, with Profile-Aware Dual-Signal Reasoning (PADR) for cold-start users.

**Key metrics:** Deployed on Tencent's Chinese news platform · LLM distillation reduces need for manual intent labels · grounded recommendations guaranteed via fuzzy SID matching

→ Full details: [papers/intent-sid-rec.md](papers/intent-sid-rec.md)

---

### 3. RuleSafe-VL (score: 74) — STRONG

**Story arc:** VLM safety benchmarks reduce moderation to label matching, revealing nothing about whether a model applies policy rules correctly → RuleSafe-VL formalizes 93 atomic rules + 92 typed relations from real platform policies into 2,166 expert-reviewed image-text cases across 3 high-risk content families to test rule-governed reasoning.

**Key metrics:** 93 rules · 92 relations · 2,166 cases · 3 content families (nudity, dangerous behavior, graphic injury) · authors: Beijing U of Posts and Telecom

→ Full details: [papers/rulesafe-vl.md](papers/rulesafe-vl.md)

---

### 4. jina-embeddings-v5-omni (score: 69) — WEAK

**Story arc:** Adding new modalities to text embedding models typically requires full retraining → frozen-tower composition freezes pretrained text encoder + separate media encoders, training only small connectors (0.35% of weights) to unify text, image, video, audio in one semantic space.

**Key metrics:** v5-omni-small 1.6B · best open-weight omni model < 2B · 0.35% trainable params · announced May 11, 2026

→ Full details: [papers/jina-embeddings-v5-omni.md](papers/jina-embeddings-v5-omni.md)

---

### 5. DRIP-R (score: 68) — STRONG

**Story arc:** LLM agent benchmarks assume unambiguous, well-specified policies — leaving a critical gap for real-world settings where policies admit multiple valid interpretations → DRIP-R systematically exploits real retail policy ambiguities to create scenarios with no single correct resolution, stress-testing agent judgment under genuine ambiguity.

**Key metrics:** Real retail policies · multiple valid interpretations · authors: IBM/Microsoft/TCD team

→ Full details: [papers/drip-r.md](papers/drip-r.md)

---

### 6. Video Understanding Reward Modeling (score: 67) — WEAK

**Story arc:** Video reward modeling severely lacks robust evaluation benchmarks and high-quality preference data → VURB benchmark (2,100 expert preference pairs with CoT traces) + VUP-35K automated pipeline + VideoDRM/VideoGRM models trained on this data.

**Key metrics:** VURB: 2,100 pairs · avg 1,143 tokens CoT per pair · VUP-35K automated dataset · 3 task types (general, long, reasoning video)

→ Full details: [papers/video-reward-model.md](papers/video-reward-model.md)
