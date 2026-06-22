# Daily Paper Inspection — 2026-06-21 (GMT+8)

> **Schedule**: 08:30 GMT+8 · **Domain**: E-commerce content ecosystem & influencer (达人) governance
> **Note**: 2026-06-21 is a **Sunday**. arXiv does not publish new listings on weekends; the most recent visible batch is from Thursday 2026-06-18. This report covers papers discovered from that batch plus all Tier 1-4 sources attempted below.

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/date/2026-06-21 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog (low prio) | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | API | yes | 403 Forbidden | 0 |
| 机器之心 jiqizhixin | wechat/WebSearch | yes | WebSearch fallback; no new Jun-21 papers | 0 |
| 新智元 | wechat/WebSearch | yes | WebSearch fallback; old articles only | 0 |
| OpenAI News | WebSearch | yes | WebSearch fallback; no domain papers | 0 |
| DeepSeek | WebSearch | yes | WebSearch fallback; general results | 0 |
| 小红书/REDtech | WebSearch | yes | WebSearch fallback; historical articles | 0 |
| Kuaishou Tech | WebSearch | yes | WebSearch fallback; Keye-VL (Nov 2025) | 0 |
| Tencent Hunyuan | WebSearch | yes | WebSearch fallback; general model info | 0 |
| arXiv 2606 WebSearch fallback | WebSearch | yes | 200 OK (via search) | 8 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **Note on network access**: All WebFetch calls to arXiv, HuggingFace, and lab blogs returned HTTP 403 in this sandbox environment. All Tier 1-3 sources were attempted; Tier 2 Semantic Scholar API also returned 403. Discovery relied entirely on WebSearch fallback for arXiv 2606 papers.

---

## Selected Papers (2026-06-18 batch — most recent before Sunday Jun 21)

All papers below have arXiv IDs starting with `2606.2xxxx`, submitted Thu 2026-06-18.

| # | Paper | Score | Bucket | Reproduce |
|---|-------|-------|--------|-----------|
| 1 | G2Rec: Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation | **81** | STRONG | `code/G2Rec/` |
| 2 | TimeProVe: Propose, then Verify for Efficient Long Video Temporal Reasoning | **80** | STRONG | `TimeProVe/` (prior run) |
| 3 | NEST: Narrative Event Structures in Time for Long Video Understanding | **70** | WEAK | — |
| 4 | Connect the Dots (CoD): Long-Lifecycle Agents with Cross-Domain Generalization | **70** | WEAK | — |
| 5 | ELVA: Exploring Ranking-Driven Universal Multimodal Retrieval | **68** | WEAK | — |
| 6 | Multi-Agent Transactive Memory (MATM) | **68** | WEAK | — |
| 7 | StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs | **64** | WEAK | — |
| 8 | NAMESAKES: Probing Identity Memorization in Text-to-Image Models | **63** | WEAK | — |

---

## Paper Details

### 1. G2Rec (Score: 81) — STRONG ★★★

**arXiv**: https://arxiv.org/abs/2606.20554  
**Authors**: Ruizhong Qiu, Yinglong Xia, Dongqi Fu, Hanqing Zeng, Ren Chen, Xiangjun Fan, Hong Li, Hong Yan, Hanghang Tong  
**Affiliations**: University of Illinois Urbana-Champaign; Meta  
**Submitted**: 2026-06-18  

**Story arc**: Existing generative recommendation methods treat item tokenization as a heuristic step disconnected from user behavior graphs, leading to suboptimal semantic representations. G2Rec unifies holistic graph-based co-engagement modeling with semantically supervised tokenization, enabling industrial-scale models to capture user interest prototypes without ground-truth labels.

**Method**: G2Rec has two pillars: (1) a scalable graph module that captures holistic user co-engagement patterns beyond local neighborhoods (avoiding GNN scalability pitfalls), and (2) a tokenization component that receives explicit supervision signals derived from graph-inferred user interest prototypes rather than relying on heuristics. Combined with an LLM-based sequential recommendation backbone, the framework was deployed online at Meta.

**Key metrics**: Online A/B deployment across Meta product surfaces confirms superiority over baselines; competitive on public sequential recommendation benchmarks.

**Innovation vs. prior**: Prior work (TIGER, ActionPiece, CoFiRec) either tokenizes with heuristics lacking supervision or uses GNNs that scale poorly. G2Rec's graph is scalable (holistic co-engagement, not local GNN) and provides supervision for tokenization—a clean combination with plausible gains.

**Score breakdown**: Innovation 23 | SOTA delta 12 | Quality 12 | Efficiency 7 | Generalization 5 | Relevance 22 = **81**

---

### 2. TimeProVe (Score: 80) — STRONG ★★★

*(Reproduction from prior run — see `TimeProVe/` folder and prior papers.json entry.)*

**arXiv**: https://arxiv.org/abs/2606.20561  
**Story arc**: Dense VLM processing over hour-long videos is cost-prohibitive. TimeProVe proposes a propose-then-verify hybrid: lightweight modules propose candidate answer+evidence windows, then an expensive VLM verifies only those windows, reducing VLM calls by 75% while improving accuracy.

---

### 3. ELVA (Score: 68) — WEAK

**arXiv**: https://arxiv.org/abs/2606.20280  
**Authors**: (from Meta / academic collaboration)  
**Submitted**: 2026-06-18  

**Story arc**: Contrastive learning for Universal Multimodal Retrieval (UMR) treats all negatives as binary (positive/negative), ignoring the graded informativeness of negatives. This causes "grain blindness" — models miss fine-grained details in complex queries. ELVA uses rule-based RL to rank negatives by similarity to positive, training the retriever to learn distinct grain information from each negative level.

**Score breakdown**: Innovation 21 | SOTA delta 10 | Quality 10 | Efficiency 7 | Generalization 3 | Relevance 17 = **68**

---

### 4–8. (See existing papers.json)

NEST (70), CoD (70), MATM (68), StylisticBias (64), NAMESAKES (63) — detailed in `papers.json`, reproduction in `TimeProVe/` for TimeProVe only.

---

## Feishu Cards

See `feishu_cards.md` for all papers with score ≥ 40.

## Code Reproduction

See `code/G2Rec/` for G2Rec (score 81).
