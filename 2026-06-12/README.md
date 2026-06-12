# 2026-06-12 电商内容生态 & 达人治理 — Paper 巡检

本日巡检窗口：**GMT+8 2026-06-12 00:00–23:59**（对应 **UTC 2026-06-11 16:00–2026-06-12 15:59**）。主要从 arXiv RSS（cs.IR/cs.CL/cs.CV/cs.AI/cs.LG）与 HuggingFace Daily Papers 拉取近 24h 更新，并按“电商内容理解/推荐/检索/去重/治理 + LLM/VLM/RAG/数据”关键词二次筛选，跳过安全/后门方向论文。

本日共整理 7 篇候选论文，评分与摘要见 `papers.json`。

## 建议关键词池（可每日固定订阅 + 二次过滤）

强相关（电商内容生态&达人治理）：e-commerce / shopping、recommendation / recsys、ranking / rerank、retrieval / vector database / ANN / ANNS、dedup / near-duplicate、content understanding、caption / multimodal、clustering / similarity、query rewrite、violation detection / moderation / policy、治理 / 审核 / 质量、data cleaning / dedup / labeling。

弱相关但值得关注（大模型方向）：LLM/VLM/MLLM、agent / tool-use、RAG、distillation、efficient inference（KV cache / flash kernel / routing）、robustness / corruption。

> 说明：与“达人治理”强相关的论文常出现在 moderation / abuse / policy / integrity / quality / fraud 方向；同时也可关注“内容质量/去重/检索”基础设施类论文（对治理成本与命中率有长期收益）。

## 推荐补充的 paper sources（便于每日抓新）

- 论文聚合：HuggingFace Daily Papers、Papers with Code（Trending / Latest）、arXiv-sanity、Connected Papers。
- arXiv 入口：订阅 cs.IR/cs.CL/cs.CV/cs.LG/cs.AI 的 daily listing（或 RSS），再用关键词二次过滤。
- 会议/评审：OpenReview（ICLR/NeurIPS/ICML/ACL 等）按 recent submissions / decisions；会议 accepted list 页面（CVPR/ICCV/ECCV、KDD/SIGIR、SIGMOD/VLDB）。
- 学术检索：Semantic Scholar（Latest + topic alerts）、Google Scholar Alerts、OpenAlex（concept + date filter）、DBLP（会议/期刊增量）。
- 工业界产出：Google Research/DeepMind、Meta AI、Microsoft Research、Amazon Science、NVIDIA Research、ByteDance Seed、Qwen、Hunyuan 等官方 blog/RSS。
- 社交媒体：X（按 research lab 账号 + trending papers）、机器之心/量子位/新智元等公众号（用于补充“热度很高”的弱相关论文线索）。

## 评分机制（100 分）

- 方法创新性：30
- 实验指标：15
- 实验质量：15
- 方法效率：10
- 方法泛化性：5
- 论文相关性：25

## 80+ 高质量论文复现

- OneRetrieval（2606.13533）：已在 `2026-06-12/OneRetrieval` 提供可运行 toy pipeline（数据/模型/训练/测试），复现“多槽位 KAE 离散码 + reserved slots + serving bypass”的关键思想。

> 说明：以上复现以“**关键机制可跑通 + 最小闭环**”为目标，使用合成/简化数据验证核心思想；与论文的真实数据、训练规模与线上系统仍有差距。
