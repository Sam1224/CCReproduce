# Feishu Cards — 2026-05-12

以下为当日评分 ≥ 40 的论文飞书卡片（共 6 篇）。

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis（Fastino AI）
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：将 LLM 安全审核还原为判别式任务：把安全任务定义、14 个危害类别和 11 种越狱策略编码为结构化 token schema 前缀，输入 0.3B 双向编码器（GLiNER2 架构），一次前向传播完成所有维度的安全分类，无需自回归生成。
💡 创新性分析：首个将 schema 前缀 + 双向编码器引入 LLM 安全审核的工作，相较 7B–27B 自回归守卫模型，参数压缩 23–90×，吞吐提升 16×，延迟降低 17×，F1 差距 ≤ 1.7 分。
📊 关键指标：9 个安全基准 · 0.3B 参数 · 16× 吞吐 · 17× 延迟降低 vs. 7B 基线（综合评分 86/100）

---

📄 标题：Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation
👥 作者：Hongyang Su, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li, Chenyun Yu（腾讯 + 中山大学）
🔗 链接：https://arxiv.org/abs/2605.07613
📝 方法概述：针对对话式推荐中隐性意图无法用关键词检索解决的问题，提出意图驱动 SID 生成框架：LLM 将 6 类用户意图映射为层次化 Semantic ID 前缀，再通过模糊匹配定位当前新闻池中的真实文章，两阶段训练结合 GPT-4 CoT 蒸馏，Profile-Aware 双信号推理解决冷启动。
💡 创新性分析：Generate-then-Match 范式将意图-SID-文章三级映射解耦，既保证推荐可落地（Grounded），又通过 GPT-4 蒸馏降低标注成本。已在腾讯新闻平台内部试点。
📊 关键指标：6 类意图（5 类隐性）· GPT-4 CoT 蒸馏 · 冷启动覆盖 · 腾讯内部部署（综合评分 76/100）

---

📄 标题：RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation
👥 作者：Zhifeng Lu, Dianyuan Wang, Yuhu Shang, Zhenbo Xu（北京邮电大学）
🔗 链接：https://arxiv.org/abs/2605.07760
📝 方法概述：现有 VLM 安全基准仅测最终标签，无法评估模型是否真正理解平台规则。RuleSafe-VL 从公开平台政策中形式化提取 93 条原子规则和 92 种类型关系，构建 2,166 条专家审核的图文案例，覆盖裸露/危险行为/血腥三大高风险类别，评测规则激活识别和规则交互推理能力。
💡 创新性分析：首个将平台审核政策规则体系形式化为可评测结构的 VLM 基准；揭示现有 SOTA 模型在规则条件推理上的显著缺口，对平台内容治理建设有直接借鉴价值。
📊 关键指标：93 条原子规则 · 92 种关系 · 2,166 专家案例 · 3 类内容（综合评分 74/100）

---

📄 标题：jina-embeddings-v5-omni: Text-Geometry-Preserving Multimodal Embeddings via Frozen-Tower Composition
👥 作者：Jina AI 团队
🔗 链接：https://arxiv.org/abs/2605.08384
📝 方法概述：冻结预训练文本编码器和各媒体编码器（图像/视频/音频），仅训练 0.35% 参数的小型连接器，将文本、图像、视频、音频统一到同一语义嵌入空间。提供 v5-omni-nano (0.9B) 和 v5-omni-small (1.6B) 两款模型。
💡 创新性分析：冻结塔融合方法极低的训练成本（仅 0.35% 参数更新）实现了多模态统一，且完整保留文本嵌入几何结构。v5-omni-small 是目前最优的 <2B 开源全模态嵌入模型，可直接用于电商商品跨模态检索和内容向量化。
📊 关键指标：0.35% 可训练参数 · 1.6B 最优 <2B 全模态 · 支持文本/图像/视频/音频（综合评分 69/100）

---

📄 标题：DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain
👥 作者：Hsuvas Borkakoty, Sebastian Pohl, Cheng Wang, Bei Chen, Yufang Hou（IBM Research / Microsoft / TCD）
🔗 链接：https://arxiv.org/abs/2605.07699
📝 方法概述：现有 Agent 基准假设策略无歧义，而真实零售/电商场景中业务政策普遍存在多种合理解读。DRIP-R 从真实零售政策文件中系统提取歧义点，构建无单一正确答案的决策场景，评测 LLM Agent 在政策歧义下的决策一致性、推理质量和歧义识别能力。
💡 创新性分析：首个专门针对零售政策歧义的 Agent 评测基准，揭示主流 LLM 在面对政策空白时倾向于武断给出确定性答案而非识别歧义的系统性缺陷。对平台规则解释 Agent、智能客服和达人治理场景有直接参考价值。
📊 关键指标：真实零售政策来源 · 多 LLM 对比评测 · 政策歧义场景覆盖多零售子域（综合评分 68/100）

---

📄 标题：Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models
👥 作者：Yuancheng Wei, Linli Yao, Lei Li, Haojie Zhang, Hao Zhou, Fandong Meng, Xu Sun（北京大学 + 腾讯微信 AI）
🔗 链接：https://arxiv.org/abs/2605.07872
📝 方法概述：针对视频理解奖励建模缺乏基准和数据的问题，提出三件套：(1) VURB 基准（2,100 偏好对 + 平均 1,143 token CoT 推理 + 多数投票）；(2) VUP-35K 全自动化偏好数据集；(3) 基于此训练的 VideoDRM 和 VideoGRM 奖励模型，在 VURB 上达到 SOTA。
💡 创新性分析：首个带有 CoT 推理痕迹的视频理解奖励基准；全自动化 VUP-35K 数据流水线为视频 MLLM 的 RLHF 训练提供了实用的大规模监督信号，可迁移到电商短视频和直播内容的自动质量评估。
📊 关键指标：VURB 2,100 对 · 平均 CoT 1,143 tokens · VUP-35K 自动化 · 3 类视频任务（综合评分 67/100）
