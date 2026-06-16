# Feishu Cards — 2026-06-15 Paper Inspection

> 仅含评分 ≥ 40 的论文，共 10 篇。

---

📄 标题：OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model
👥 作者：Xuxin Zhang, Ben Chen, Yue Lv 等（Kuaishou Technology）
🔗 链接：https://arxiv.org/abs/2606.13533
📝 方法概述：用单一生成式检索模型统一电商多路召回，引入 KAE（关键词对齐编码）+ reserved slots + 推理期 bypass，实现小时级新词注入且无需重训。多阶段 SFT 保证深召回质量，线上 A/B 逐步替换各召回分支。
💡 创新性分析：将"运营可编辑性"提升为生成式召回的一等约束，reserved slot 绑定使新词路由变为可控接口，是工程与算法的双重创新。
📊 关键指标：离线 Order HR@350=0.5482；线上 A/B：仅替换倒排分支 Order +0.710%（显著），扩展全部分支 CTR +0.821%（显著）。评分：87/100

---

📄 标题：Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations
👥 作者：Zhuohang Jiang, Yuxin Chen, Shijie Wang 等（Kuaishou Technology）
🔗 链接：https://arxiv.org/abs/2606.10357
📝 方法概述：AIR 将 LLM 推理离线化，把用户跨域行为序列分解为"意图原子"并建索引；在线服务只做轻量原子检索与组合，实现 ~400× 推理加速，同时保持语义一致性。解决内容平台→电商跨域推荐的语义鸿沟与延迟双重挑战。
💡 创新性分析：将 LLM 语义能力解耦为"离线生产 + 在线检索组合"，是工业 LLM 部署的重要模式创新；意图原子化提供了细粒度、可组合的语义单元。
📊 关键指标：多公开 dataset 达 SOTA；快手电商生产 A/B 测试 GMV +3.446%，延迟 ~400× 降低。评分：87/100

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang 等（Alibaba International Digital Commerce Group）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：将电商查询推荐重构为 memory-augmented 智能体框架，通过链式检索优化使 agent 在生成 query 后立即召回真实库存商品并迭代精炼；设计一致性奖励联合优化 query 相关性与下游商品参与度（CVR）。
💡 创新性分析：首次将 chain-of-retrieval 嵌入电商 query 推荐 RL 训练，从结构上对齐 CTR（query相关性）与 CVR（商品转化），解决电商搜索中的 CTR-CVR 脱钩问题。
📊 关键指标：Industrial Cons@1: 0.025→0.117 (+368%)；生产 A/B：Query CTR +2.9%，Guided CVR +3.1%。评分：86/100

---

📄 标题：Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants
👥 作者：Shuxian Fan, Seonwoo Min, Youna Hu 等（Amazon）
🔗 链接：https://arxiv.org/abs/2606.12608
📝 方法概述：专家编写多轮购物助手评测基准，将能力拆分为 5 类推理/15 子类，并为每个任务提供可判定的二值 rubric（必需项/加分项）。使用经一致性验证的 LLM-as-judge 规模化评测，输出按重要性加权的通过率。
💡 创新性分析：把开放式对话评测转为条款级可复核的通过率指标，让诊断短板成为可能；LLM judge 对齐专家标注（F1=0.749），自动评测可落地。
📊 关键指标：525 missions，10,863 rubrics（85% 必需）；Judge macro-F1=0.749，κ=0.498；主流模型通过率 57.4%–77.2%。评分：82/100

---

📄 标题：CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning
👥 作者：Penghui Yang, Long Xing, Xiaoyi Dong 等（Shanghai AI Lab / InternLM / USTC / CUHK）
🔗 链接：https://arxiv.org/abs/2606.09393
📝 方法概述：将 RLVR 引入密集图像/视频描述任务，以"纯语言模型能否凭 caption 回答 MCQ"为奖励信号，完全无需人工参考标注。解耦的两阶段 pipeline（VLM 生成 → LLM 评分）使奖励可验证且可扩展。
💡 创新性分析：将 caption 质量重定义为信息效用而非文本相似度，RLVR 奖励完全自动化，紧凑型 3B 模型性能超越 Qwen2.5-VL-72B，适用于商品图描述/短视频内容理解等电商场景。
📊 关键指标：20+ 图像/视频基准提升；3B/7B 模型性能 ≥ Qwen2.5-VL-72B 和 Qwen3-VL-235B-A22B。评分：82/100

---

📄 标题：CFALR: Collaborative Filtering-Augmented Large Language Model for Personalized Fashion Outfit Recommendation
👥 作者：Yujuan Ding, Junrong Liao, Yunshan Ma 等（PolyU / SMU / NUS 等）
🔗 链接：https://arxiv.org/abs/2606.13001
📝 方法概述：将穿搭推荐重构为 LLM 生成式任务（P-FITB + outfit generation），把 CF embedding 与多模态特征通过投影层注入为非文本 token 输入 LLM，并在输出侧融合 CF 信号与 LLM logits，兼顾语义理解与个性化。
💡 创新性分析：将协同过滤信号结构化对齐到 LLM 空间，解决组合推荐的语义-协同错位，是 CF-LLM 融合的优雅实现。
📊 关键指标：P-FITB Acc（Polyvore 1/4=0.6498；IQON 1/4=0.6103）；生成指标 PP/OC/MPC（Polyvore 0.2428/0.4626/0.3556）。评分：77/100

---

📄 标题：Gaze Heads: How VLMs Look at What They Describe
👥 作者：Rohit Gandikota, David Bau（MIT Bau Lab）
🔗 链接：https://arxiv.org/abs/2606.14703
📝 方法概述：发现 VLM 语言骨干中存在少量 attention heads（gaze heads）专门追踪当前描述的视觉区域；通过推理期重定向这些 heads 的注意力，可强制模型描述指定区域，无需重训。
💡 创新性分析：定位了具备因果控制力的功能性 head 集合，为 VLM 内容理解可控性与可解释性提供了推理期干预接口。
📊 关键指标：对 top-100 gaze heads（<9%）一次干预，描述重定向准确率 83.1%；在 COCO 图像上也有效。评分：71/100

---

📄 标题：TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation
👥 作者：Ziheng Chen, Jiali Cheng, Zezhong Fan 等
🔗 链接：https://arxiv.org/abs/2606.07688
📝 方法概述：研究生成式推荐中的概念遗忘（unlearning）问题。语义 ID（SID）被 forget/retain item 共享，直接抑制导致副作用。TRACER 将概念相关 item 重新分配到替代 token，并用 coherence regularizer 保护 retain 集合的语义一致性。
💡 创新性分析：从 token 共享角度分析 unlearning 冲突，token reassignment 从结构上解决问题，与推荐系统合规/隐私治理直接相关。
📊 关键指标：比基线更好地移除目标概念同时保留推荐效用（具体数值见论文表格）。评分：70/100

---

📄 标题：Generative Archetype-Grounded Item Representations for Sequential Recommendation
👥 作者：Yifan Li, Jiahong Liu, Xinni Zhang 等
🔗 链接：https://arxiv.org/abs/2606.11023
📝 方法概述：GenAIR 用 LLM 从 item metadata 推断"archetype"（理想受众画像），将 archetype embedding 注入序列推荐模型，并加入行为校准目标把语义空间拉回真实交互分布。
💡 创新性分析：用受众画像（谁会买）而非物品属性（是什么）表示 item，行为校准保证语义与行为对齐，思路清晰。
📊 关键指标：3 个真实数据集均显著优于 SOTA；开源代码。评分：69/100

---

📄 标题：When Recommendation Denoising Meets Popularity Bias: Understanding and Mitigating Their Interaction
👥 作者：Guohang Zeng, Jie Lu, Guangquan Zhang
🔗 链接：https://arxiv.org/abs/2606.14046
📝 方法概述：发现 loss-based 去噪训练会把"难学但干净"的长尾样本误作噪声降权，放大流行度偏置。PAD（Popularity-Aware Denoising）用物品流行度构造门控系数，对头部商品强去噪、对尾部保守，作为轻量 plug-in。
💡 创新性分析：形式化刻画去噪与流行度偏置的耦合机制，PAD gate 即插即用，理论贡献清晰。
📊 关键指标：在准确性与多样性联合指标上优于 vanilla 去噪基线（具体数值见论文表格）。评分：56/100
