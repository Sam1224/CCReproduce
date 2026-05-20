# Daily AI Paper Inspection — 2026-05-19 (GMT+8)

**Domain:** E-commerce Content Ecosystem & Influencer (达人) Governance  
**Emphasis:** LLM / VLM / MLLM, Captioning, Vector Embeddings, Clustering & Similarity, Agents, RAG, Distillation, Violation Detection, Content Governance, Data Quality, Large-Scale Data Labeling

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| huggingface.co/papers/date/2026-05-19 | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| huggingface.co/papers/trending | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| deepmind.google/discover/blog/ | blog | yes | 403 Forbidden | 0 |
| research.google/blog/ | blog | yes | 403 Forbidden | 0 |
| ai.meta.com/research/ | blog | yes | 403 Forbidden | 0 |
| anthropic.com/news | blog | yes | 403 Forbidden | 0 |
| seed.bytedance.com/research | blog | yes | 403 Forbidden | 0 |
| tech.meituan.com/ | blog | yes | 403 Forbidden | 0 |
| qwenlm.github.io/blog/ | blog | yes | 403 Forbidden | 0 |
| qbitai.com/ | wechat | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| WebSearch: arxiv 2605.XXXXX domain papers | aggregator (fallback) | yes | 200 | 5 |
| WebSearch: arxiv Jan-Apr 2026 domain papers | aggregator (fallback) | yes | 200 | 4 |
| WebSearch: 机器之心 jiqizhixin.com 2026-05 | wechat | yes | no relevant hits | 0 |
| WebSearch: 新智元 mp.weixin.qq.com 2026-05 | wechat | yes | no relevant hits | 0 |
| WebSearch: OpenAI news May 2026 | blog | yes | 200 (product news, no research papers) | 0 |
| WebSearch: DeepSeek May 2026 | blog | yes | 200 (no new May 2026 model paper) | 0 |
| WebSearch: 小红书 REDtech LLM 2026-05 | wechat | yes | 200 (no new May 2026 papers indexed) | 0 |
| WebSearch: Kuaishou 快手 LLM 2026-05 | wechat | yes | 200 (older papers, no new May 2026) | 0 |
| WebSearch: Tencent Hunyuan 2026-05 | wechat | yes | 200 (no new May 2026 papers) | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **Note:** arXiv direct URL access returned 403 from this environment. All arXiv papers were discovered via WebSearch (Tier 3 fallback). Papers with IDs 2605.XXXXX are from May 2026; TIGER-FG (2605.18434) has a high ID consistent with indexing on ~May 18-19, 2026. Papers from 2601/2602 are high-relevance domain finds included via "pastweek catch-up" logic.

---

## Selected Papers

| # | Title | arXiv ID | Score | Bucket | Code |
|---|-------|----------|-------|--------|------|
| 1 | When Rules Fall Short: Agent-Driven Discovery of Emerging Content Issues in Short Video Platforms | 2601.11634 | **89** | STRONG | `code/AgentIssueDiscovery/` |
| 2 | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs | 2602.08355 | **84** | STRONG | `code/EVAds/` |
| 3 | TIGER-FG: Text-Guided Implicit Fine-Grained Grounding for E-commerce Retrieval | 2605.18434 | **83** | STRONG | `TIGER-FG/` (pre-existing) |
| 4 | CVA: Compressed Video Aggregator for Efficient Micro-Video Recommendation | 2605.08810 | **79** | STRONG | — |
| 5 | AuDisAgent: Training-Free Multimodal Controversy Detection | 2605.02939 | **76** | STRONG | — |
| 6 | Thinking Broad, Acting Fast: Latent Reasoning Distillation for E-Commerce Relevance | 2601.21611 | **75** | STRONG | — |
| 7 | MAP-V: Transparent and Controllable Recommendation Filtering via Multimodal Multi-Agent | 2604.17459 | **72** | STRONG | — |
| 8 | GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion | 2605.00670 | **70** | STRONG | — |
| 9 | Who Decides What Is Harmful? Content Moderation Policy via Multi-Agent Personalised Inference | 2605.01416 | **61** | WEAK | — |

---

## Paper Summaries

### 1. When Rules Fall Short (Score: 89) ⭐ TOP PICK
- **Link:** https://arxiv.org/abs/2601.11634
- **Affiliation:** TikTok Inc.
- **Domain relevance:** Short video platform content governance, LLM agent-driven clustering, policy iteration
- See full summary: `papers/when-rules-fall-short.md`
- Code: `code/AgentIssueDiscovery/`

### 2. E-VAds (Score: 84) ⭐
- **Link:** https://arxiv.org/abs/2602.08355
- **Affiliation:** Alibaba (Taobao)
- **Domain relevance:** E-commerce short video understanding benchmark, commercial intent reasoning
- See full summary: `papers/e-vads.md`
- Code: `code/EVAds/`

### 3. TIGER-FG (Score: 83) ⭐
- **Link:** https://arxiv.org/abs/2605.18434
- **Domain relevance:** E-commerce image retrieval, text-guided fine-grained visual grounding, dual distillation
- See full summary: `papers/tiger-fg.md`
- Code: `TIGER-FG/` (pre-existing reproduction)

### 4. CVA (Score: 79)
- **Link:** https://arxiv.org/abs/2605.08810
- **Domain relevance:** Micro-video recommendation, content-driven embeddings, efficiency
- See full summary: `papers/cva-video-rec.md`

### 5. AuDisAgent (Score: 76)
- **Link:** https://arxiv.org/abs/2605.02939
- **Domain relevance:** Social video platform content governance, multimodal controversy detection
- See full summary: `papers/audisagent.md`

### 6. Thinking Broad, Acting Fast (Score: 75)
- **Link:** https://arxiv.org/abs/2601.21611
- **Domain relevance:** E-commerce search relevance, CoT distillation, latent reasoning
- See full summary: `papers/thinking-broad-acting-fast.md`

### 7. MAP-V (Score: 72)
- **Link:** https://arxiv.org/abs/2604.17459
- **Domain relevance:** Recommendation filtering, multimodal multi-agent, preference graph
- See full summary: `papers/map-v.md`

### 8. GRE-MC (Score: 70)
- **Link:** https://arxiv.org/abs/2605.00670
- **Domain relevance:** Multimodal recommendation, missing modality completion, graph retrieval
- See full summary: `papers/gre-mc.md`

### 9. Who Decides What Is Harmful (Score: 61)
- **Link:** https://arxiv.org/abs/2605.01416
- **Domain relevance:** Content moderation, multi-agent LLM, personalized safety
- See full summary: `papers/who-decides-harmful.md`
