# 2026-06-10 电商内容生态 & 达人治理 — Paper 巡检

本日共整理 9 篇候选论文（均为 2026-06-09 近 24 小时 arXiv 更新，按与电商内容理解/检索/治理的相关性与热度筛选）。

## 推荐补充的 paper sources（便于每日抓新）
- 论文聚合：Papers with Code（Trending / Latest）、arXiv-sanity、Connected Papers。
- arXiv 入口：除关键词检索外，建议固定订阅 cs.IR/cs.CL/cs.CV/cs.LG 的 daily listing（或 RSS），再用关键词二次过滤。
- 会议/评审：OpenReview（ICLR/NeurIPS/ICML/ACL 等）按「最近提交/最新决定」筛选；会议 accepted list 页面（CVPR/ICCV/ECCV、KDD/SIGIR、SIGMOD/VLDB）。
- 学术检索：Semantic Scholar（Latest + topic alerts）、Google Scholar Alerts、OpenAlex（concept + date filter）。
- 工业界产出：Google Research/DeepMind、Meta AI、Microsoft Research、Amazon Science、NVIDIA Research、ByteDance Seed、Qwen、Hunyuan 等官方 blog/RSS。

## 评分机制（100 分）
- 方法创新性：30
- 实验指标：15
- 实验质量：15
- 方法效率：10
- 方法泛化性：5
- 论文相关性：25

评分明细见 `papers.json`。

## 80+ 高质量论文复现
- SuperFashion（2606.10697）：已在 `2026-06-10/SuperFashion` 提供可运行的 toy pipeline（数据/模型/训练/测试），用于复现其“superpixel tokenization + attribute-conditioned retrieval”的关键思想。

> 说明：本复现以可跑通的最小闭环为目标，使用合成服饰 toy 数据集验证“超像素 token 更利于属性特定检索”的直觉与效果；与原论文的真实数据集与完整工程实现仍有差距。
