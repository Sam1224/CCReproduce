# 2026-09-17 电商内容生态 & 达人治理 Paper 巡检

## 巡检口径
- **时间窗口**：2026-09-17 00:00:00 ~ 23:59:59（GMT+8），对应 UTC 2026-09-16T16:00:00Z ~ 2026-09-17T15:59:59Z。
- **检索策略**：优先用 arXiv-first 流程检索当日/近两日相关论文与热门增量，再补 Hugging Face Daily Papers、Scholar / Semantic Scholar / OpenAlex / DBLP、AI lab blog、GitHub 官方仓库、公开 benchmark / accepted 页面做交叉核验。
- **关键词修正**：在用户给定关键词基础上，补充了 **long-sequence recommendation、sponsored search relevance、shopping agents、case-driven relevance optimization、LLM annotation cascade、progressive curriculum、agentic visual workflows、programmatic video understanding**。
- **排除规则**：继续排除安全 / 后门 / 越狱 / 投毒 / 对抗防御相关论文，不纳入今日名单。

## 今日结论
今天真正**同时满足“新鲜度 + 电商强相关 + 方法质量”**的候选并不多，因此最终名单分成两层：
1. **强相关高价值主名单**：直接服务于电商广告推荐、赞助检索、搜索相关性治理、购物 agent 评测；
2. **弱相关高热补位**：与内容生态/达人生产链路弱相关，但对多模态创作基础设施与 agent 能力评估非常值得跟踪。

最终保留 6 篇：
- **TM20K（93）**—— 今天最值得复现的强相关工业论文之一，直指电商广告推荐里的超长行为序列建模与效率平衡。
- **Scaling Dense Retrieval with LLM-Annotated Training Data（92）**—— Walmart 赞助搜索 dense retrieval 生产系统，数据构造、标注与课程学习都很实。
- **Case-Driven Multi-Agent Framework（89）**—— ByteDance 电商搜索相关性闭环治理，把 bad-case、标注、优化和 memory 串成 agent 系统。
- **RecoAtlas（78）**—— 购物 recommendation agent benchmark 很有方法论价值，但与直接线上收益的距离略远，且官方 GitHub 为空仓库。
- **LynnReal-Omni（76）**—— 今日高热多模态视频生成系统，对创作者工具链和虚拟达人内容生产有启发。
- **BVB（74）**—— agentic video understanding benchmark，适合作为多模态 agent 评测方法补充。

## 评分机制（百分制）
- **方法创新性**：30 分
- **实验指标**：15 分
- **实验质量**：15 分
- **方法效率**：10 分
- **方法泛化性**：5 分
- **论文相关性**：25 分

### 评分原则说明
- **相关性** 优先看它是否直接服务于电商内容理解、达人/商品治理、推荐/检索、广告、数据质量、评测治理或 shopping agent。
- **创新性** 更看系统性改造，不给“小修小补”过高分。
- **实验质量** 重点看是否有清晰 baseline、消融、线上或跨场景证据。
- **效率** 单独评分，因为电商搜索/推荐/内容链路高度实时、长序列且成本敏感。

## 复现结论（评分 >= 80）
本次达到复现阈值的共有 3 篇：

1. **TM20K（93）**
   - 今日未发现可信的公开官方代码实现。
   - 已在 `2026-09-17/TM20K/` 中补充 toy-but-runnable PyTorch 复现，覆盖 teacher / token-merged student / distillation / baseline。

2. **Scaling Dense Retrieval with LLM-Annotated Training Data（92）**
   - 今日未发现可信的公开官方代码实现。
   - 已在 `2026-09-17/StructuredCurriculumDR/` 中补充 toy-but-runnable PyTorch 复现，覆盖 structured mining / annotation cascade / BCE→MNR→Triplet curriculum。

3. **Case-Driven Multi-Agent Framework（89）**
   - 今日未发现可信的公开官方代码实现。
   - 已在 `2026-09-17/CaseDrivenSearchAgents/` 中补充 toy-but-runnable PyTorch 复现，覆盖 User / Annotator / Optimizer Agent 与 Global Memory 闭环。

### 其余论文
- **RecoAtlas（78）**：论文给出了 GitHub 链接，但仓库当前为空；由于今日评分未到 80，本次不做重复实现。
- **LynnReal-Omni（76）**：已核验官方 GitHub 仓库非空，包含实际推理与工程目录，因此不重复复现。
- **BVB（74）**：已核验官方 GitHub 仓库非空，包含 benchmark / sandbox / eval 代码，因此不重复复现。

## Web App 与数据库
今日 `papers.json` 已补齐：
- 中英文方法概览 / 故事线 / 创新点 / 关键指标；
- 分项评分与评分依据；
- 标签、日期、官方代码 / 复现链接；
- `figure_steps` 与 `exp_cards`，用于自动生成 methodology 图和实验亮点图。

Web 展示将继续沿用浅色 research style，并支持：
- 中英文切换；
- 日期筛选；
- 最低分滑动条过滤；
- methodology / experiment 图显示隐藏；
- SQLite + JSON 双份结果归档。

## 备注
- 今天最值得业务团队认真看的 3 篇，分别对应 **长序列广告推荐、赞助搜索 dense retrieval、电商搜索 relevance 治理闭环**。
- 今天的“弱相关高热补位”主要集中在 **agentic visual workflow** 与 **programmatic video understanding benchmark** 两个方向。
- 若要真正实现 **每日 00:00 自动执行**，还需要在外部持久化触发器/工作流中注册定时任务；仓库脚本本身已经具备被调度的基础。 
