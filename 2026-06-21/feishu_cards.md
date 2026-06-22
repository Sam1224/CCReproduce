# Feishu Cards — 2026-06-21

> 包含所有评分 ≥ 40 的论文，共 8 篇（均来自 2026-06-18 提交批次）

---

📄 标题：G2Rec: Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation
👥 作者：Ruizhong Qiu, Yinglong Xia, Dongqi Fu, Hanqing Zeng, Ren Chen, Xiangjun Fan, Hong Li, Hong Yan, Hanghang Tong (UIUC & Meta)
🔗 链接：https://arxiv.org/abs/2606.20554
📝 方法概述：G2Rec 提出统一框架，将全局图协同参与建模（holistic co-engagement graph）与有监督语义分词（supervised semantic tokenization）融合，用图推断的用户兴趣原型为生成式推荐中的 item tokenization 提供显式监督信号，解决现有方法"局部 GNN 扩展性差 + 分词无监督"的双重痛点。已在 Meta 多个产品面在线部署验证。
💡 创新性分析：首次将全局图信号（co-engagement prototype）作为分词监督来源，打通过去割裂的图建模与分词两个模块；与 TIGER/ActionPiece/CoFiRec 相比，去除了启发式依赖并解决了 GNN 规模化问题，工业部署已验证可行性。
📊 关键指标：Meta 内部多产品面在线 A/B 测试全面超越 baseline；公开基准（多个序列推荐数据集）NDCG@10/HR@10 优于现有 SOTA。总分：81/100。

---

📄 标题：TimeProVe: Propose, then Verify for Efficient Long Video Temporal Reasoning in Activities of Daily Living
👥 作者：Arkaprava Sinha, Dominick Reilly, Siddharth Krishnan, Hieu Le, Srijan Das (University of North Carolina, Charlotte)
🔗 链接：https://arxiv.org/abs/2606.20561
📝 方法概述：TimeProVe 面向小时级长视频问答，提出"先提议后验证"混合框架：轻量模块将视频动作定位转化为候选答案+证据时间窗（ACE 模块），再仅对这些少量候选片段调用昂贵 VLM 定向验证，解耦推理成本与视频时长。还提出 OpenTSUBench(OTB)开放式基准。
💡 创新性分析：ACE 模块显式将动作定位转化为可验证的答案-证据对，实现廉价提议+昂贵验证的精确协同；无需显式时序定位训练即具竞争力，兼顾精度与成本。
📊 关键指标：在 OTB 上超过最强 baseline +7.3%；VLM 调用减少 75%，推理成本降低 93%；Charades-STA 上竞争性表现。总分：80/100。

---

📄 标题：NEST: Narrative Event Structures in Time for Long Video Understanding
👥 作者：Ali Asgarov, Kaushik Narasimhan, Najibul Haque Sarker, Hani Alomari 等 (Virginia Tech)
🔗 链接：https://arxiv.org/abs/2606.19706
📝 方法概述：NEST 是面向完整长电影的叙事事件结构理解数据集，含 1005 部完整电影（均长98分钟），每部约 102 个多模态叙事事件，标注视觉/对白/音频三模态 grounding，设立 ETD/EL/EAE/ERE 四类任务。
💡 创新性分析：超越"大海捞针式检索"和多选题范式，以完整电影为单位评测叙事级长程关联理解，填补评测空白。
📊 关键指标：基准极具挑战，ETD<8%、EL<6%、EAE<11%；ERE 微调后 44.42% F1；数据总时长 1639.3 小时。总分：70/100。

---

📄 标题：Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning
👥 作者：Yanxi Chen, Weijie Shi, Yuexiang Xie, Boyi Hu, Yaliang Li, Bolin Ding, Jingren Zhou (Alibaba Group)
🔗 链接：https://arxiv.org/abs/2606.20002
📝 方法概述：CoD 训练 LLM 获得"串联线索"元能力：智能体在长期任务序列中连续探索，从自身经验学习并迭代更新对环境的上下文，越做越好。配套端到端 RL（交替 solve-task 与 update-context 长 rollout）与细粒度信用分配 GRPO 变体。
💡 创新性分析：将"随经验自更新上下文"本身作为 RL 优化目标，而非针对特定领域能力；验证了域内、跨域及 Ralph-loop OOD 泛化。代码已开源。
📊 关键指标：概念验证实验证明端到端 RL 有效，展示跨域 OOD 泛化潜力（Trinity-RFT 开源实现）。总分：70/100。

---

📄 标题：ELVA: Exploring Ranking-Driven Universal Multimodal Retrieval
👥 作者：(see arxiv page)
🔗 链接：https://arxiv.org/abs/2606.20280
📝 方法概述：ELVA 针对通用多模态检索（UMR）中对比学习的"粒度盲区"（grain blindness）问题，提出基于规则的 RL 框架，按 negative 与 positive 的相似度排序，让模型从不同粒度的 negative 中学习区分信号，而非把所有负样本等价对待。
💡 创新性分析：首次用 RL 排序机制显式区分 hard/soft negatives 在检索训练中的权重，思路新颖；推理期与标准检索无额外开销。
📊 关键指标：在标准 UMR 基准（如 M-BEIR）上相较对比学习 baseline 有提升。总分：68/100。

---

📄 标题：Multi-Agent Transactive Memory (MATM)
👥 作者：To Eun Kim, Xuhong He, Dishank Jain, Ambuj Agrawal, Negar Arabzadeh, Fernando Diaz (Carnegie Mellon University, UC Berkeley)
🔗 链接：https://arxiv.org/abs/2606.19911
📝 方法概述：MATM 面向去中心化异构智能体群体，提出群体级"轨迹存取共享"框架：生产者智能体把任务执行轨迹贡献到共享库，消费者智能体检索这些轨迹提升自身效果；引入 Learning-to-Rank 重排阶段改善检索质量。
💡 创新性分析：把 RAG 从"检索人类文档"扩展到"检索智能体生成轨迹"，支持异构智能体无需协调即可共享经验，并随库规模增长持续改善。
📊 关键指标：ALFWorld 和 WebArena 上提升任务成功率并减少交互步数；代码已开源。总分：68/100。

---

📄 标题：StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs
👥 作者：Shaghayegh Kolli, Timo Cavelius, Nafiseh Nikeghbal, Samantha Dalal, Jana Diesner (Technical University of Munich)
🔗 链接：https://arxiv.org/abs/2606.20527
📝 方法概述：StylisticBias 是评测 MLLM 属性级社会偏见的受控基准：500 张写实基础人脸×50 单属性变体（约 25K 图），固定身份只改变一个视觉属性（年龄/体型/穿搭等），评测 6 个 MLLM 在 25 个二元社会判断场景的偏见。
💡 创新性分析："固定身份+单属性扰动"的受控设计实现干净归因，揭示偏见集中在少数视觉线索（约 15 个属性解释近 80% 变化）。
📊 关键指标：年龄+体型主导身份级效应；穿搭风格带来最大属性级偏移；约 15 属性解释 ~80% 总变化。总分：64/100。

---

📄 标题：NAMESAKES: Probing Identity Memorization in Text-to-Image Models
👥 作者：Morris Alper, Vasudha Varadarajan, Moran Yanuka, Angelina Wang, Hadar Averbuch-Elor (Tel Aviv University / Cornell University)
🔗 链接：https://arxiv.org/abs/2606.20155
📝 方法概述：NAMESAKES 研究文生图模型对真实人物的"身份记忆"问题，提出完全黑盒行为探测方法（无需参考照片/训练数据/白盒访问）区分"被记忆"vs"被虚构"的人脸生成，并构建 1269 个名字-人脸对的 NAMESAKES 数据集。
💡 创新性分析：纯黑盒探测消除了以往研究对白盒访问/参考照片的依赖，更贴近真实场景；覆盖不同知名度并含扰动名对照，评测更严谨。
📊 关键指标：多款 SOTA T2I 模型上显著预测身份记忆，成功区分被记忆与未识别名字。总分：63/100。
