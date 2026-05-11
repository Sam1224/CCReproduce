# UniVA: Unified Value Alignment for Generative Recommendation in Industrial Advertising

- **arXiv**: https://arxiv.org/abs/2605.05803
- **作者**: Xinxun Zhang, Yuling Xiong, Jiale Zhou, Zhengkai Guo, Zhennan Pang, Junbang Huo 等 16 人
- **机构**: Tencent (WeChat Channels 广告平台)
- **领域标签**: `Generative Recommendation` `LLM-Recommender` `Advertising` `RL` `电商-广告`
- **桶位**: STRONG

## 方法概述
UniVA 把生成式推荐 (Generative Recommendation, GR) 在工业广告场景下的核心痛点抽象为 **value-aware 解码** 问题：传统 GR 基于 next-token 生成商品 SID，但只关注语义 (用户兴趣)，对商业价值 (eCPM/GMV) 缺乏端到端对齐。

UniVA 由三个组件串成统一的 value alignment 流水线：(1) **Commercial SID Tokenizer** 把价值属性 (商品出价、品类毛利等) 注入到 SID 构造，使 SID 表征本身就具备 value-discriminative 能力；(2) **Generation-as-Ranking SID Decoder** 同时受 SFT 和 eCPM-aware 强化学习监督，把价值得分融进 next-item SID 生成步骤，将"生成"和"排序"折叠成一次解码；(3) **Value-guided Personalized Beam Search** 复用前述 logits 作为在线价值引导，并用个性化 trie tree 约束解码到合法 SID 路径。

## 故事
现有生成式推荐管线 → **以语义为中心、未对齐商业价值** → 在工业广告中导致 ranking 阶段二次重排序、训练在线不一致。
UniVA 在 tokenization / decoding / serving 三个层面同时注入价值信号，使生成-排序-在线对齐到一致目标。

## 创新性分析
- 把"value alignment"从经典 reward-shaping 上升到 **token 表示** 层面 (Commercial SID)，是 GR 与 RecSys-RL 路线一个新颖的切入点。
- **Generation-as-Ranking** 单次解码完成生成+排序，把 GR 离线/在线 gap 折叠掉，工程价值非常高。
- 个性化 trie 约束 + value beam search 的组合是工业落地的关键 (避免合规和 GMV 损失)。
- 与 TIGER、HSTU 等纯语义 GR 工作正交，与 SimRec / Lurec 之类排序蒸馏路线互补。

## 关键指标
- **Tencent WeChat Channels 广告平台离线**: Hit Rate@100 提升 **+37.04%**.
- **在线 A/B**: GMV **+1.5%** lift。
- (论文未公开 Recall@k、CTR、CPM 拆解；仅给出综合提升)

## 评分 (满分 100)
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 24 / 30 | 三个组件的组合是新的；Commercial SID 是真正的新点 |
| 实验指标 | 14 / 15 | 离线 +37% Hit@100、在线 +1.5% GMV，工业级数字 |
| 实验质量 | 13 / 15 | 在生产平台 A/B；缺少跨平台 / 跨域 ablation |
| 效率 | 7 / 10 | 解码代价由 trie 控制，但 RL 训练成本不低 |
| 泛化性 | 3 / 5 | 仅在 WeChat Channels 实验，但方法论可迁移 |
| 相关性 | 25 / 25 | 工业广告生成式推荐 — 电商核心场景 |
| **合计** | **86** | ✅ ≥80，进入复现 |

## 复现状态
- 论文未提供代码。
- 仓库 `code/UniVA/` 包含一个忠实于原文结构的 toy PyTorch 实现：Commercial SID tokenizer / Generation-as-Ranking decoder / value-guided beam search 三件套，附 toy 数据 + 训练 + 推理脚本。
