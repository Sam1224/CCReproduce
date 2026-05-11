# Daily AI Paper Inspection — 2026-05-10

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Run time:** 08:30 GMT+8 (00:30 UTC)  
**Yesterday (GMT+8):** 2026-05-10  
**arXiv target prefix:** 2605.XXXXX (May 2026)

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|---|---|---|---|---|
| arXiv cs.CL /list/new | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| arXiv cs.CV /list/new | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| arXiv cs.IR /list/new | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| arXiv cs.MM /list/new | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| arXiv cs.LG /list/new | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| arXiv cs.CL monthly listing 2026-05 | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| HuggingFace Daily Papers 2026-05-10 | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| Semantic Scholar API | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| OpenAlex API | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| DBLP search API | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| Papers With Code API | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| arXiv-sanity-lite | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| OpenReview API | aggregator | yes | 403 Host not in allowlist | 0 (web-search fallback used) |
| Web search fallback (targeted 2605 queries) | aggregator (fallback) | yes | 200 | 28 candidates |
| Google AI blog | blog | yes | 403 Host not in allowlist | 0 (indirect via search) |
| DeepMind blog | blog | yes | 403 Host not in allowlist | 0 (indirect via search) |
| Meta AI blog | blog | yes | 403 Host not in allowlist | 0 (indirect via search) |
| OpenAI news | blog | yes | 403 Host not in allowlist | 0 (indirect: Codex safety May 8, GPT-5.5-Cyber May 7) |
| Anthropic news | blog | yes | 403 Host not in allowlist | 0 (indirect: AI for Science May 7) |
| DeepSeek | blog | yes | 403 Host not in allowlist | 0 (indirect: "Thinking with Visual Primitives" ~May 5) |
| Qwen research | blog | yes | 403 Host not in allowlist | 0 (indirect: Qwen-Scope May 1, Qwen3Guard) |
| Tencent Hunyuan | blog | yes | 403 Host not in allowlist | 0 (no relevant posts found) |
| Xiaohongshu/RedTech | blog | yes | 403 Host not in allowlist | 0 (no public blog found) |
| Meituan tech | blog | yes | 403 Host not in allowlist | 1 (AI Coding with Agent eval May 7) |
| Kuaishou tech | blog | yes | 403 Host not in allowlist | 0 (no relevant posts found) |
| ByteDance research | blog | yes | ECONNREFUSED | 0 (indirect: MixFormer recommender) |
| Facebook Research blog | blog | yes | 403 Host not in allowlist | 0 (covered under Meta AI) |
| ICML/NeurIPS/ICLR/CVPR/ACL accepted lists | conf | yes | 403 / not yet published for 2026 | 0 |
| KDD/SIGIR/RecSys/ACMMM 2026 | conf | yes | 403 / not yet published | 0 |

> **Note:** All direct API and web-fetch calls returned HTTP 403 (sandbox egress restricted to allowlisted hosts). Papers were discovered through targeted Google Web Search queries using site:arxiv.org/abs/2605 patterns and keyword searches. The 28 candidates above include papers from the 2605.00xxx–2605.07xxx range, submitted May 1–10, 2026, filtered for domain relevance.

---

## Candidate Papers

### STRONG bucket (directly ecom / content / influencer governance / violation detection / data labeling)

| # | Title | arXiv ID | Score | Bucket |
|---|---|---|---|---|
| 1 | EKTM: Effective Knowledge Transfer for Multi-Task Recommendation | 2605.05730 | **82** | STRONG |
| 2 | AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce | 2604.20135 | **81** | STRONG |
| 3 | LiSCP: Lightweight Stylistic Consistency Profiling for Multimedia Moderation | 2605.05950 | **77** | STRONG |
| 4 | VANGUARD: Reasoning-Guided Grounding for Video Anomaly Detection via MLLMs | 2605.02912 | **71** | STRONG |
| 5 | PREFER: Personalized Review Summarization with Online Preference Learning | 2605.05911 | **68** | STRONG |
| 6 | Who Decides What Is Harmful? Multi-Agent Personalised Content Moderation | 2605.01416 | **64** | STRONG |

### WEAK bucket (high-impact LLM/VLM, transferable to ecom/governance)

| # | Title | arXiv ID | Score | Bucket |
|---|---|---|---|---|
| 7 | LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG | 2605.06285 | **78** | WEAK |
| 8 | EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer | 2605.01732 | **61** | WEAK |

---

## Code Reproduction

Papers with score ≥ 80 receive full PyTorch reproduction:
- `code/EKTM/` — multi-task recommendation knowledge transfer (score 82)
- `code/AFMRL/` — attribute-enhanced fine-grained multimodal e-commerce representation (score 81)

---

## Paper Detail Files

- [`papers/ektm.md`](papers/ektm.md)
- [`papers/afmrl.md`](papers/afmrl.md)
- [`papers/liscp.md`](papers/liscp.md)
- [`papers/vanguard.md`](papers/vanguard.md)
- [`papers/prefer.md`](papers/prefer.md)
- [`papers/who_decides_harmful.md`](papers/who_decides_harmful.md)
- [`papers/latentrag.md`](papers/latentrag.md)
- [`papers/egad.md`](papers/egad.md)

## Feishu Cards

→ [`feishu_cards.md`](feishu_cards.md) (all papers with score ≥ 40)
