# 2026-06-07 Daily AI Paper Inspection
**Domain:** E-commerce Content Ecosystem & Influencer (达人) Governance
**GMT+8 Date:** 2026-06-07 | **Run Time:** 08:30 GMT+8

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|---------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| HuggingFace papers/date/2026-06-07 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | API | yes | 403 Forbidden | 0 |
| 机器之心 (WebSearch fallback) | wechat | yes | 200 (search) | 0 directly (older articles indexed) |
| 新智元 (WebSearch fallback) | wechat | yes | 200 (search) | 0 directly |
| OpenAI News (WebSearch fallback) | blog | yes | 200 (search) | 0 directly |
| DeepSeek (WebSearch fallback) | blog | yes | 200 (search) | 0 directly |
| Kuaishou Tech (WebSearch fallback) | blog | yes | 200 (search) | 1 (OneReason arXiv:2606.06260) |
| Tencent Hunyuan (WebSearch fallback) | blog | yes | 200 (search) | 0 directly |
| Xiaohongshu / RedTech (WebSearch fallback) | blog | yes | 200 (search) | 0 directly |
| arXiv 2606.* WebSearch (Tier-3 fallback) | aggregator | yes | 200 (search) | 8 candidates discovered |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

**Discovery note:** arXiv listing pages and HuggingFace returned HTTP 403 for all direct WebFetch calls. All paper discovery relied on targeted WebSearch over 2606.XXXXX arXiv IDs with domain-focused query terms. Eight unique papers were found; two additional papers (QueryAgent-R1, AITDNA) were added beyond the prior-session baseline.

---

## Selected Papers — Sorted by Score

| Rank | Paper | ArXiv ID | Score | Bucket | Tags |
|------|-------|----------|-------|--------|------|
| 🥇 | [QueryAgent-R1](papers/query_agent_r1.md) | 2606.05671 | **82** | STRONG | e-commerce, agent, RL, retrieval |
| 🥇 | [OpAI-Bench](papers/opai_bench.md) | 2606.06481 | **82** | STRONG | AIGC-detection, content-governance |
| 🥈 | [AdaPlanBench](papers/adaplan_bench.md) | 2606.05622 | **80** | WEAK→MEDIUM | agent, planning, benchmark |
| 🥉 | [OneReason](papers/one_reason.md) | 2606.06260 | **79** | STRONG | e-commerce, rec-sys, RL |
| 🥉 | [camroll-agent](papers/camroll_agent.md) | 2606.05275 | **79** | WEAK | VLM, agent, retrieval |
| — | [Code2LoRA](papers/code2lora.md) | 2606.06492 | **72** | WEAK | code, PEFT, hypernetwork |
| — | [Your AI Text is not Mine (AITDNA)](papers/aitdna.md) | 2606.04906 | **70** | MEDIUM | AIGC-detection, benchmark |
| — | [Stance Simulation Audit](papers/stance_simulation_audit.md) | 2606.06443 | **61** | WEAK | stance, audit, multimodal |

---

## Score Rubric

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

## Code Reproduction (score ≥ 80)

| Paper | Score | Official Code | Reproduction |
|-------|-------|--------------|--------------|
| QueryAgent-R1 | 82 | None found | ✅ Reproduced → `code/QueryAgentR1/` |
| OpAI-Bench | 82 | [VILA-Lab/OpAI-Bench](https://github.com/VILA-Lab/OpAI-Bench) | Authors provided; see link |
| AdaPlanBench | 80 | [JiayuJeff/AdaPlanBench](https://github.com/JiayuJeff/AdaPlanBench) | Authors provided; see link |