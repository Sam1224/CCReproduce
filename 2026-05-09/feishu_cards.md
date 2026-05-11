# Feishu Cards — 2026-05-09 Daily AI Paper Inspection
# 飞书卡片 — 2026-05-09 日报 (分数 ≥ 40 的论文)

---

📄 标题：ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring
👥 作者：Deyi Ji, Junyu Lu, Xuanyi Liu, Liqun Liu, Hailong Zhang et al.
🔗 链接：https://arxiv.org/abs/2605.02200
📝 方法概述：ARGUS 提出三阶段政策自适应广告治理框架，专门解决平台政策频繁迭代导致历史标注失效的核心问题。第一阶段 Policy Seeding 建立新政策基础感知；第二阶段引入"检察官-辩护人-裁判官"（PDU）对抗三智能体架构，结合 RAG 增强的政策知识库自动修正历史偏差标签，为 RL 提供高质量奖励信号；第三阶段通过合成灰色地带对抗样本持续精化模型推理能力。全流程 VLM + RL + RAG 有机结合，实现政策涌现时的快速自适应。
💡 创新性分析：首次将"检察官-辩护人-裁判官"对抗辩证范式引入广告治理标注修正，突破传统静态分类器无法适应政策动态变化的瓶颈；PDU 架构自动处理新旧策政策标签冲突，无需人工重新标注；VLM+RL+RAG 三者协同的闭环设计在工业级广告治理场景首次验证。评分：84/100（STRONG）
📊 关键指标：三阶段递增消融验证每阶段贡献显著；灰色地带违规 Historical Recall 持续提升（Stage I→II→III 清晰递增）；在线广告合规数据集上整体 F1 达 SOTA；直接适用于电商达人营销内容审核、广告违规检测场景。

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：GLiGuard 将 LLM 安全护栏重新设计为模式条件化多维度分类问题，以 GLiNER2 双向编码器（0.3B 参数）为骨干，通过 schema 输入同时评估 prompt 安全性、response 安全性、拒绝检测、14 类细粒度伤害类别和 11 类越狱策略，单次前向传播完成全部 25 个安全维度评估，彻底避免自回归解码器的顺序生成成本。
💡 创新性分析：将 LLM 护栏从"生成问题"逆转为"分类问题"，0.3B 编码器在 9 个标准安全基准上达到与 7B–27B 解码器竞争的 F1 分数；参数量缩小 23–90 倍的同时实现 16× 吞吐量提升和 17× 延迟降低；schema 条件化设计允许灵活组合安全维度，无需重新设计 prompt，工程实用性极高。评分：80/100（STRONG）
📊 关键指标：9 个标准 LLM 安全基准 F1 与 SOTA 竞争；推理延迟降低 17×（实测）；吞吐量提升 16×；模型仅 0.3B 参数；适用于电商平台实时 LLM 输出审核、UGC 内容安全过滤。

---

📄 标题：DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain
👥 作者：Hsuvas Borkakoty, Sebastian Pohl, Cheng Wang, Bei Chen, Yufang Hou
🔗 链接：https://arxiv.org/abs/2605.07699
📝 方法概述：DRIP-R 是首个专门针对零售域真实政策歧义的 LLM Agent 评测基准。构建精心设计的政策歧义场景（退货/换货/会员权益等），配套多样化客户人物画像，通过全双工多轮工具调用对话模拟，采用四维多裁判评估框架（政策遵从性、对话质量、行为一致性、解决方案质量）系统评估 LLM Agent 在政策模糊情境下的决策能力。
💡 创新性分析：首次系统利用零售政策歧义构建 Agent 评测基准，填补现有基准假设政策明确的关键空白；关键发现：前沿 LLM（GPT-5、Gemini 等）对相同政策歧义场景存在根本性裁决分歧，证明歧义是真实且系统性的挑战；多裁判四维评估框架比单指标评测更接近真实部署需求。评分：76/100（STRONG）
📊 关键指标：多个前沿 LLM 在相同歧义场景下裁决分歧显著（无一致正确答案）；对话质量和行为一致性跨模型差异大；基准涵盖真实零售退货/换货/会员政策场景，直接适用于电商客服 Agent 部署风险评估。

---

📄 标题：Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation
👥 作者：Siyuan Li, Aodu Wulianghai, Xi Lin, Xibin Yuan, Qinghua Mao, Guangyan Li, Xiang Chen, Jun Wu, Jianhua Li
🔗 链接：https://arxiv.org/abs/2605.05950
📝 方法概述：LiSCP 提出基于风格一致性档案的轻量级 AI 生成文本检测方法。核心思路：对输入文本生成多个与配套图像/视频语义对齐的改写变体，构建融合离散风格信号（词汇多样性、句式复杂度）和连续语义信号（跨变体语义对齐、与多媒体上下文隐式对齐）的一致性档案，利用人类写作和 LLM 生成文本在改写后的风格一致性模式上的本质差异进行分类。
💡 创新性分析：将检测目标从"具体文本特征"转移到"行为一致性档案"，从根本上规避被改写攻击绕过的风险；多模态对齐改写确保跨领域泛化；跨领域检测提升 11.79%，在对抗改写场景下鲁棒性显著优于先前方法。评分：75/100（STRONG）
📊 关键指标：跨领域检测准确率提升 11.79%（vs SOTA 检测器）；多媒体新闻/电影字幕数据集域内检测 SOTA；对抗改写场景下准确率保持率显著高于基线；直接适用于电商平台虚假评论检测、AI 代笔内容识别。

---

📄 标题：Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework
👥 作者：Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak（波兰科学院等）
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：提出个性化多智能体内容审核推断框架，通过专家智能体（领域专属伤害分析）、管理智能体（编排专家选择）、幽灵档案智能体（模拟目标用户敏感性视角）三层协同，将中心化"一刀切"审核转换为面向用户个性化敏感性档案的差异化推断。相同内容对不同用户群体输出差异化审核建议。
💡 创新性分析：幽灵档案智能体创造性地将"受众视角"引入审核决策，首次系统解决审核中的主观性差异问题；32% 准确率提升证明个性化对审核质量的实质性影响。评分：66/100（WEAK）
📊 关键指标：与用户敏感性对齐准确率提升 32%（vs 中心化规则基线）；适用于电商平台多元用户群的差异化内容策略制定。

---

📄 标题：Multimodal Data Curation Through Ranked Retrieval
👥 作者：Pratyush Muthukumar et al.（NVIDIA）
🔗 链接：https://arxiv.org/abs/2605.01163
📝 方法概述：NVIDIA 提出双组件多模态数据策划框架：Symmetric Nucleus Subsampling（SNS）从训练对两侧裁剪到最能相互支撑的部分消除标注噪声，Expert Embedding Engine（EEE）通过学习投影网络组合多个互补嵌入专家并引入偏差感知目标减少模态驱动的聚类分离，将嵌入空间模态间隙平均缩小 90%+。
💡 创新性分析：SNS+EEE 同时作用于数据和模型侧联合优化，90%+ 模态间隙消弭在多异构数据集上验证，对电商图文跨模态检索质量有直接提升潜力。评分：69/100（WEAK）
📊 关键指标：多异构数据集模态间隙缩小 >90%；跨模态检索 Recall@K 显著提升；NVIDIA 工业级验证可信度高。

---

📄 标题：The Cost of Context: Mitigating Textual Bias in Multimodal Retrieval-Augmented Generation
👥 作者：Hoin Jung, Xiaoqian Wang（Purdue University）
🔗 链接：https://arxiv.org/abs/2605.05594
📝 方法概述：揭示 MLLM+RAG 系统中的"重腐败（Recorruption）"现象：即便引入完全准确的 oracle 上下文，也会导致 MLLM 放弃最初正确预测。通过注意力矩阵机制诊断，发现重腐败由"视觉盲目"（视觉注意力被压制）和"结构性位置偏置"（优先关注边界 token）驱动，提出 attention re-calibration 缓解方案。
💡 创新性分析："重腐败"概念首次定义量化，揭示 RAG"成功幻觉"（位置巧合而非真实理解），机制性注意力诊断对优化多模态 RAG 系统有重要指导价值。评分：64/100（WEAK）
📊 关键指标：重腐败现象在多个 MLLM 上验证；attention re-calibration 后重腐败率显著降低；适用于电商商品多模态知识库问答系统优化。
