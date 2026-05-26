# Daily AI Paper Inspection — 2026-05-25 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Scheduled run:** 08:30 GMT+8 · Powered by Claude Code  
**Discovery window:** Papers published/indexed during GMT+8 calendar day 2026-05-25

---

## Source Coverage

> **Note:** arXiv listing pages, HuggingFace aggregators, and most lab blogs returned HTTP 403 from this sandbox environment. WebSearch was used as the primary fallback. Papers from May 2026 (arXiv ID prefix `2605.`) were located via targeted WebSearch queries; the very freshest May 25 submissions may not yet be indexed by search engines as of run time.

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 3) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 4) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 2) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 1) |
| arxiv cs.LG/new | aggregator | no | skipped after repeated 403s; WebSearch used | 1 |
| arxiv cs.AI/new | aggregator | no | skipped after repeated 403s; WebSearch used | 0 |
| huggingface.co/papers/date/2026-05-25 | aggregator | yes | 403 Forbidden | 0 |
| huggingface.co/papers/trending | aggregator | yes | 403 Forbidden | 0 |
| deepmind.google/discover/blog | blog | yes | 403 Forbidden | 0 |
| research.google/blog | blog | yes | 403 Forbidden | 0 |
| ai.meta.com/research | blog | yes | 403 Forbidden | 0 |
| anthropic.com/news | blog | yes | 403 Forbidden | 0 |
| seed.bytedance.com/research | blog | yes | 403 Forbidden | 0 |
| tech.meituan.com | blog | no | not attempted (repeated 403 pattern) | 0 |
| qwenlm.github.io/blog | blog | yes | 403 Forbidden | 0 |
| qbitai.com (量子位) | wechat | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| 机器之心 jiqizhixin.com (WebSearch) | wechat | yes | partial results | 0 new |
| 新智元 mp.weixin.qq.com (WebSearch) | wechat | no | not specifically queried | 0 |
| OpenAI News (WebSearch) | blog | yes | searched; no May 25 papers | 0 |
| DeepSeek (WebSearch) | blog | yes | searched; no new May 2026 papers | 0 |
| Xiaohongshu / REDtech (WebSearch) | wechat | no | not attempted (no results pattern) | 0 |
| Kuaishou Tech (WebSearch) | blog | yes | searched; no May 25 specific | 0 |
| Tencent Hunyuan (WebSearch) | blog | yes | searched; no May 25 papers | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

**Summary:** 17 sources attempted · 0 direct WebFetch successes (all 403) · 8 papers discovered via WebSearch across `2605.*` arXiv IDs.

---

## Picked Papers

| # | Slug | Title | arXiv | Bucket | Score |
|---|------|-------|-------|--------|-------|
| 1 | [vina_aigc](#vina) | VINA: Video as Natural Augmentation — Unified AI-Generated Image & Video Detection | 2605.21977 | STRONG | **83** |
| 2 | [gliguard](#gliguard) | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | 2605.07982 | STRONG | **77** |
| 3 | [tgqformer_ecom](#tgqformer) | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation | 2605.17366 | STRONG | **76** |
| 4 | [latentprobe_nsfw](#latentprobe) | Latent Space Probing for Adult Content Detection in Video Generative Models | 2605.00874 | STRONG | **72** |
| 5 | [grmcrec](#grmcrec) | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion | 2605.00670 | STRONG | **69** |
| 6 | [granurag](#granurag) | From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG | 2605.15019 | WEAK | **57** |
| 7 | [llava_ckd](#llava-ckd) | LLaVA-CKD: Bottom-Up Cascaded Knowledge Distillation for Vision-Language Models | 2605.10641 | WEAK | **55** |
| 8 | [repair_rag](#repair-rag) | Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization | 2605.18772 | WEAK | **51** |

---

## Code Reproductions (score ≥ 80)

| Paper | Score | Code folder |
|-------|-------|-------------|
| VINA | 83 | `code/VINA/` |

---

<a name="vina"></a>
## 1. VINA — Unified AIGC Detection (Score 83)
→ Full summary: [`papers/vina_aigc.md`](papers/vina_aigc.md)  
→ Code: [`code/VINA/`](code/VINA/)

<a name="gliguard"></a>
## 2. GLiGuard — LLM Safeguard (Score 77)
→ Full summary: [`papers/gliguard.md`](papers/gliguard.md)

<a name="tgqformer"></a>
## 3. TGQ-Former — E-Commerce Recommendation (Score 76)
→ Full summary: [`papers/tgqformer_ecom.md`](papers/tgqformer_ecom.md)

<a name="latentprobe"></a>
## 4. LatentProbe — Adult Content Detection (Score 72)
→ Full summary: [`papers/latentprobe_nsfw.md`](papers/latentprobe_nsfw.md)

<a name="grmcrec"></a>
## 5. GRMCRec — Modality Completion (Score 69)
→ Full summary: [`papers/grmcrec.md`](papers/grmcrec.md)

<a name="granurag"></a>
## 6. GranuRAG — Multi-Granularity RAG (Score 57)
→ Full summary: [`papers/granurag.md`](papers/granurag.md)

<a name="llava-ckd"></a>
## 7. LLaVA-CKD — Cascaded KD (Score 55)
→ Full summary: [`papers/llava_ckd.md`](papers/llava_ckd.md)

<a name="repair-rag"></a>
## 8. RePAIR — RAG Error Correction (Score 51)
→ Full summary: [`papers/repair_rag.md`](papers/repair_rag.md)
