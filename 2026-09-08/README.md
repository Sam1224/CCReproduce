# 2026-09-08 电商内容生态 & 达人治理 Paper 巡检

## 巡检口径
- **执行日期**：2026-09-08（GMT+8）。
- **执行窗口**：2026-09-08 00:00:00 ~ 23:59:59（GMT+8），对应 UTC 2026-09-07T16:00:00Z ~ 2026-09-08T15:59:59Z。
- **检索策略**：优先扫 arXiv recent/new（cs.IR / cs.LG / cs.CV / cs.AI / cs.CL），再用 Hugging Face Daily Papers、OpenReview / accepted 信息、Google Scholar / Semantic Scholar / OpenAlex / DBLP 做补充核验，并回查 GitHub / project page / lab blog 公开线索。
- **关键词修正**：在用户给定关键词基础上，额外补充了 **multimodal Doc2Query、agentic recommendation memory、entity-level visual retrieval、partially relevant video retrieval、autonomous launch review、long-context agent memory** 等更贴近内容电商链路的主题。
- **排除规则**：继续跳过安全 / 后门 / 越狱 / 投毒 / guardrail 对抗方向，以及以 exploit / red-team 为主要卖点的论文。

## 今日结论
今天严格按“**强相关优先 + 弱相关高热补位**”筛选后，主名单保留 7 篇：
1. **SAM-D2Q（85）**：今天最值得跟进，直接对应电商商品内容理解、图文对齐与搜索分发。
2. **AtomRec（79）**：对内容推荐与长期兴趣建模很有前瞻性，但生产证据还不够强。
3. **Distill Globally, Adapt Locally（78）**：很强的推荐蒸馏路线，适合大规模降本增效。
4. **SAGE（79）**：商品详情长图 / 参数图理解的高价值论文，和电商视觉内容非常贴近。
5. **ITA（77）**：对直播回放检索、长视频巡检和帧级证据召回很有价值。
6. **AutoLR（74）**：今天最值得关注的弱相关高热论文之一，代表工业推荐研发自动化。
7. **KVMem（72）**：优秀的 agent infra 补位，弱相关但很值得持续关注。

> 说明：本轮公开平台里，真正同时满足“今日可见 / 强相关 / 质量足够高”的论文数量并不多，因此保留了两篇高热基础设施论文做补位，但相关性分明显低于前五篇。

## 评分机制（百分制）
- **方法创新性**：30 分
- **实验指标**：15 分
- **实验质量**：15 分
- **方法效率**：10 分
- **方法泛化性**：5 分
- **论文相关性**：25 分

### 评分原则说明
- **相关性** 优先看是否能服务于电商内容理解、达人/内容推荐、审核巡检、图文检索、治理评测或数据构造链路。
- **创新性** 看是否改写了关键环节，而不是做轻微调参。
- **实验质量** 重点考察 baseline、消融、跨数据集迁移和生产证据。
- **效率** 单独打分，因为内容生态链路通常成本敏感、时延敏感。

## 复现结论（评分 >= 80）
本次达到复现阈值的共有 1 篇：

1. **SAM-D2Q（85）**
   - 今日未检到可直接使用的官方代码发布。
   - 已在 `2026-09-08/SAM-D2Q/` 中补充 toy-but-runnable PyTorch 复现，覆盖数据、模型、训练、测试完整 pipeline。

## Web App 与数据库
今日 `papers.json` 已补齐：
- 中英文方法概览 / 故事线 / 创新点 / 关键指标；
- 分项评分与评分依据；
- 论文日期、标签、复现代码链接占位；
- `figure_steps` 与 `exp_cards`，用于自动生成 methodology 图和实验亮点图。

现有 Web App 继续沿用浅色 research 风格，并支持：
- 中英文切换；
- 日期筛选；
- 最低分滑动条过滤；
- 评分明细与评分依据；
- methodology figure / experiment highlights 显示隐藏；
- SQLite + JSON 双份结果归档。

## 备注
- 今天没有把安全 / 后门类论文纳入主名单。
- SAGE 与 ITA 虽然论文页挂了代码链接，但公开仓库当前仍偏占位态，因此不算“已有可直接运行代码”。
- 若要真正做成 **每日 00:00 自动执行**，还需要把现有巡检脚本挂到可用的 workflow / trigger 上；本轮已经完成脚本与数据资产准备，自动调度部分还需要平台侧 workflow 模板支撑。 
