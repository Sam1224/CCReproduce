# 电商内容生态 & 达人治理 Paper巡检 — 2026-05-22 (GMT+8)

> 巡检窗口: 2026-05-21T16:00Z ~ 2026-05-22T16:00Z (UTC) ≡ 2026-05-22 全天 (Asia/Shanghai)
> 数据来源: WebSearch fallback (arXiv / HuggingFace 直接访问均返回 403)
> 巡检时间: 2026-05-23 00:30 UTC (scheduled daily run)

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arXiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 4) |
| arXiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 3) |
| arXiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 4) |
| arXiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 2) |
| arXiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 3) |
| arXiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 2) |
| arXiv cs.CL/pastweek | aggregator | yes | 403 Forbidden | 0 |
| arXiv cs.CV/pastweek | aggregator | yes | 403 Forbidden | 0 |
| arXiv cs.IR/pastweek | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Daily Papers 2026-05-22 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Trending Papers | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar API (2026-05-22) | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind Blog | blog | yes | 403 Forbidden | 0 |
| Google Research Blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech Blog | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (WebSearch: 2 refs) |
| 机器之心 Jiqizhixin | wechat | yes | WebSearch fallback | 1 |
| 新智元 | wechat | yes | WebSearch fallback | 0 |
| OpenAI News | blog | yes | WebSearch fallback | 0 |
| DeepSeek | blog | yes | WebSearch fallback | 0 |
| Xiaohongshu / RedTech | blog | yes | WebSearch fallback | 1 (NoteLLM prior work) |
| Kuaishou Tech | blog | yes | WebSearch fallback | 1 (QARM prior work) |
| Tencent Hunyuan | blog | yes | WebSearch fallback | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| **WebSearch arXiv 2605.* discovery** | aggregator | yes | 200 OK | **18 candidates** |

> **Note:** May 22, 2026 is a Friday. All direct web endpoints (arXiv listing pages, HuggingFace, lab blogs) returned HTTP 403 from the remote sandbox. All paper discovery used WebSearch fallback. Candidates are from arXiv May 2026 (2605.xxxxx) prefixed papers discovered via web search; exact submission dates for individual papers cannot be verified since arXiv listing pages are blocked. Papers already covered in earlier sessions (2026-05-08 through 2026-05-14) are excluded.

---

## Scoring Rubric

| Dimension | Max |
|-----------|-----|
| Innovation | 30 |
| Experimental SOTA delta | 15 |
| Experimental quality / ablations | 15 |
| Efficiency | 10 |
| Generalization | 5 |
| Domain relevance (ecom + governance) | 25 |
| **Total** | **100** |

---

## Selected Papers — Scored (6 papers)

### STRONG — Directly Ecom / Content Governance / Influencer

| # | Title | arXiv | Score | Bucket | Code Repro |
|---|-------|-------|-------|--------|-----------|
| 1 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | 2605.07982 | **82** | STRONG | Official code exists (fastino-ai/GLiGuard) |
| 2 | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation | 2605.17366 | **79** | STRONG | — |
| 3 | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems | 2605.09338 | **73** | STRONG | — |
| 4 | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework | 2605.01416 | **60** | STRONG | — |

### WEAK — High-Impact LLM/VLM, Transferable to Ecom/Governance

| # | Title | arXiv | Score | Bucket |
|---|-------|-------|-------|--------|
| 5 | Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents | 2605.21347 | **58** | WEAK |
| 6 | SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution | 2605.18401 | **55** | WEAK |

---

## Code Reproductions (Score ≥ 80)

| Paper | Model | Status |
|-------|-------|--------|
| GLiGuard (2605.07982) | GLiGuard | Official code verified at github.com/fastino-ai/GLiGuard + HuggingFace model. No toy reproduction needed. |

---

## Output Files

- `papers/GLiGuard.md` — GLiGuard detailed summary + scoring
- `papers/TGQ-Former.md` — TGQ-Former detailed summary + scoring
- `papers/MM-LLM-Rec.md` — MM-LLM Recommendation Framework summary + scoring
- `papers/Who-Decides-Harmful.md` — Multi-Agent content moderation summary + scoring
- `papers/InsightsGenerator.md` — Insights Generator summary + scoring
- `papers/SkillsVote.md` — SkillsVote summary + scoring
- `feishu_cards.md` — Feishu cards for all 6 papers (score ≥ 40)
- `web/` — Single-page static app with zh/en toggle, score slider, date picker
