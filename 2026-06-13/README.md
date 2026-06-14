# 2026-06-13 电商内容生态 & 达人治理 — Paper 巡检 (Daily AI Paper Inspection)

本日巡检窗口：**GMT+8 2026-06-13 00:00–23:59**（对应 **UTC 2026-06-12 16:00–2026-06-13 15:59**）。主要以 arXiv（SubmittedDate）为主源，结合“电商内容理解/检索/对话购物助手/治理 + LLM/VLM/Agent/RAG/数据”关键词做二次筛选，跳过安全/后门相关论文。

本日共整理 **9 篇**候选论文（含补充发现），评分与中英文摘要见 `papers.json`；单篇详情见 `papers/<slug>.md`；飞书卡片见 `feishu_cards.md`；可视化界面见 `web/index.html`。

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback→WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback→WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback→WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback→WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback→WebSearch) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback→WebSearch) |
| HuggingFace papers/date/2026-06-13 | aggregator | yes | 403 Forbidden | 44 via GitHub mirror |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (superseded by mirror) |
| HuggingFace papers GitHub mirror (AtharvaDomale) | aggregator | yes | 200 OK | 44 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes (WebSearch fallback) | n/a | 0 |
| ByteDance Seed Research | blog | yes (WebSearch fallback) | n/a | 1 (UNIVID 2606.05748) |
| Meituan Tech | blog | yes (WebSearch fallback) | n/a | 0 |
| Qwen Blog | blog | yes (WebSearch fallback) | n/a | 0 |
| 量子位 QbitAI | wechat | yes (WebFetch 403 + WebSearch) | 403 + fallback | 0 |
| 机器之心 jiqizhixin.com | wechat | yes (WebSearch) | n/a | 0 (2024–2025 results only) |
| 新智元 mp.weixin.qq.com | wechat | yes (WebSearch) | n/a | 0 |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 |
| OpenAI News | blog | yes (WebSearch) | n/a | 0 |
| DeepSeek blog | blog | yes (WebSearch) | n/a | 0 (early-2026 mHC paper only) |
| 小红书 REDtech | wechat | yes (WebSearch) | n/a | 0 (2024 results only) |
| 快手 Kuaishou Tech | blog | yes (WebSearch) | n/a | 1 (OneRetrieval 2606.13533) |
| 腾讯混元 Hunyuan | blog | yes (WebSearch) | n/a | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **注：** arXiv 直连全部返回 403；通过 HuggingFace GitHub mirror（200 OK）确认 June 13 共 44 篇，结合 WebSearch fallback 发现额外候选。所有来源均已尝试并记录真实结果，未虚构任何论文数据。

---

## 建议关键词池（可每日固定订阅 + 二次过滤）

强相关（电商内容生态&达人治理）：e-commerce / shopping、shopping assistant / conversational commerce、product retrieval / search / ranking / rerank、vector database / ANN / embedding、query understanding / rewrite、content understanding / multimodal / caption、dedup / near-duplicate / similarity / clustering、violation detection / moderation / policy compliance、governance / integrity、data attribution / data selection、data cleaning / labeling。

弱相关但值得关注（大模型方向）：LLM/VLM/MLLM、agent / tool-use / memory、RAG、distillation、efficient post-training、robustness / evaluation。

## 推荐补充的 paper sources（便于每日抓新）

- 论文聚合：HuggingFace Daily Papers、Papers with Code（Trending / Latest）、arXiv-sanity。
- arXiv 入口：cs.IR/cs.CL/cs.CV/cs.LG/cs.AI 的 daily listing / RSS（再做关键词二次过滤）。
- 会议/评审：OpenReview（ICLR/NeurIPS/ICML/ACL 等）按 recent submissions / decisions；会议 accepted list（CVPR/ICCV/ECCV、KDD/SIGIR、SIGMOD/VLDB）。
- 学术检索：Semantic Scholar（topic alerts）、Google Scholar Alerts、OpenAlex（concept + date filter）、DBLP（会议/期刊增量）。
- 工业界产出：Google Research/DeepMind、Meta AI、Microsoft Research、Amazon Science、NVIDIA Research、Qwen、Hunyuan 等官方 blog/RSS。
- 社交媒体：X（research lab 账号 + trending papers）、机器之心/量子位/新智元等公众号（用于补充“热度很高”的弱相关论文线索）。

## 评分机制（100 分）

- 方法创新性：30
- 实验指标：15
- 实验质量：15
- 方法效率：10
- 方法泛化性：5
- 论文相关性：25

## 📊 Selected Papers (by score)

| # | Title | ArXiv ID | Score | Bucket | Affiliation |
|---|-------|----------|-------|--------|-------------|
| 1 | OneRetrieval: Unifying Multi-Branch E-commerce Retrieval | 2606.13533 | **87** ★ | STRONG | Kuaishou |
| 2 | UNIVID: Unified VLM for Video Moderation | 2606.05748 | **86** ★ | STRONG | ByteDance |
| 3 | Influcoder: Distilling Gradient Influence into an Encoder | 2606.13668 | **82** ★ | STRONG | Inria |
| 4 | SkillChain: Image-Based E-Commerce AI Assistants | 2606.12984 | **78** | STRONG | Alibaba |
| 5 | Two-Agent Simulation for Agentic Search (eBay) | 2606.12924 | **78** | STRONG | eBay |
| 6 | EvoArena: Memory Evolution for Robust LLM Agents | 2606.13681 | **68** | WEAK | NUS et al. |
| 7 | RA-RFT: Reasoning by Analogy via RAG+RLVR | 2606.13680 | **67** | WEAK | various |
| 8 | Multi-Agent RL from Delayed Marketplace Feedback | 2606.13604 | **67** | WEAK | DoorDash |
| 9 | QueryAgent-R1: E-Commerce Query Recommendation | 2606.05671 | **73** | STRONG | Alibaba Intl |

★ Score ≥ 80: code reproduction available

## 80+ 高质量论文复现

- **OneRetrieval（2606.13533）**：`2026-06-13/OneRetrieval/` — 快手，生成式电商多路召回，Keyword-Aligned Encoding + 可编辑 reserved slots。
- **Influcoder（2606.13668）**：`2026-06-13/Influcoder/` — Inria，LoRA 梯度 influence 排名 → CountSketch → encoder embedding 蒸馏，数据归因/毒性过滤。
- **UNIVID（2606.05748）**：`2026-06-13/code/UNIVID/` — ByteDance，策略感知视频字幕 VLM for 内容治理（policy-aware captioning → moderation pipeline）。

> 说明：复现以”关键机制可跑通 + 最小闭环”为目标，使用合成/简化数据验证核心思想；与论文的真实数据、训练规模与线上系统仍有差距。
