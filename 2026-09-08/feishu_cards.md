# 2026-09-08 飞书巡检卡片文案

📄 标题：SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce  
👥 作者：Hui Zhou, Jian Hui Ji, Lei Ma, Rong Xiao, Xiaoyi Zeng  
🔗 链接：https://arxiv.org/abs/2609.04961  
📝 方法概述：把传统 Doc2Query 升级成电商场景的多模态离线扩写框架，用 information-gain 约束筛样本、用视觉遮蔽补商品图属性，再用业务价值做 preference alignment。最终生成的 pseudo-query 不只是“像搜索词”，而是更贴近搜索需求与转化价值。  
💡 创新性分析：创新在于把监督样本、视觉增强和业务奖励三层一起重写，非常符合工业电商搜索链路；相比纯文本 Doc2Query，更能补图像属性和高价值需求词。  
📊 关键指标：离线 Relevance 43.11%→74.96%；Top-3000 相关 item 数 454.5→594.1（+30.7%）；AliExpress Search 在线 GMV +3.38%，Pay Count +2.27%。

---

📄 标题：AtomRec: Evolving Atomic Memory for Agentic Recommendation  
👥 作者：Peiyu Hu, Weihai Lu, Siying Gu, Zhuodong Liu, Zhaokai Luo, Yuean Niu, Zhiyong Wang, Jia Wang  
🔗 链接：https://arxiv.org/abs/2609.04882  
📝 方法概述：把推荐系统的长期记忆从粗粒度 summary 升级为 atomic memory，并通过语义 link 形成多跳证据路径；新行为到来时，系统还会动态更新相关旧记忆。  
💡 创新性分析：真正的新意是把记忆做成“可原子化、可链接、可演化”的 substrate，不只是拉长上下文；这对内容兴趣演化和推荐解释性很有价值。  
📊 关键指标：四个 benchmark 平均约 +8.5% 相对提升；Books H@5 0.7764→0.8543，MovieTV H@5 0.8654→0.9364。

---

📄 标题：Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation  
👥 作者：Siliang Liu, Mohammad Ghasemi, Sapan Patel, Amin Banitalebi-Dehkordi  
🔗 链接：https://arxiv.org/abs/2609.05363  
📝 方法概述：先让 LLM teacher 为商品对生成 trade-up 关系标签与 rationale，再把 reasoning 信号蒸馏到轻量 pair classifier，最后用 product-type test-time training 做类目级适配。  
💡 创新性分析：亮点是把推理能力蒸馏成工业可用的非生成 student，并用 PT-TTT 解决不同类目“升级标准”不一致的问题。  
📊 关键指标：student AUC 0.912→0.924；加 PT-TTT 后 AUC 0.941、AP 0.940；推断约比直接 LLM 快 5000 倍、成本低 10000 倍。

---

📄 标题：SAGE: Semantic Attribute Graphs for Multi-Entity Visual Retrieval  
👥 作者：Yongjoo Kim, Mincheol Kwon, Seonga Choi, Minseung Lee, Kyeong-Jin Oh, Hyunyoung Lee, Yunsu Choi, Jungbeom Lee  
🔗 链接：https://arxiv.org/abs/2609.04255  
📝 方法概述：针对商品详情长图和参数对比图里的多实体检索问题，先解析实体和属性，再构造成层级语义图做子图级检索，从而缓解 dense image 中的 semantic dilution。  
💡 创新性分析：把检索粒度从 patch 提升到实体-属性图节点，并配套给出 DEAR benchmark；方向非常贴近商品长图内容理解。  
📊 关键指标：DEAR 上 Recall@3 = 0.8488，generation score = 2.7462；多实体比较问题上的增益最明显。

---

📄 标题：Intrinsic Temporal Adaptation of CLIP for Partially Relevant Video Retrieval  
👥 作者：Hyun Seok Seong, Woojin Jun, SuBeen Lee, Jae-Pil Heo  
🔗 链接：https://arxiv.org/abs/2609.04800  
📝 方法概述：把 temporal adaptation 内生到 CLIP backbone 末层，再用 AWGP 把弱监督训练信号扩散到多个相关帧，提升部分相关视频检索里的 moment 级证据质量。  
💡 创新性分析：不是再堆一个更重的时序头，而是先把 frame representation 变得更有时间感；对直播回放检索和内容巡检很有启发。  
📊 关键指标：QVHighlights 上 R@1 = 39.8、R@5 = 67.2、SumR = 281.9；TVR / ActivityNet / Charades-STA 的 SumR 分别为 224.7 / 201.4 / 87.4。

---

📄 标题：AutoLR: Automating the Path from Research to Launch Review in Industrial Recommender Systems  
👥 作者：Qi Zhang, Yanlin Chen, Wenchao Xiao  
🔗 链接：https://arxiv.org/abs/2609.04871  
📝 方法概述：用 multi-expert council、evidence-weighted selector 和 layered knowledge system，把推荐系统从 research 到 launch review 的长链路串成自治研发 harness。  
💡 创新性分析：强项在系统编排与工业流程自动化，不是单点模型创新；更像“受控 agent 研发流水线”。  
📊 关键指标：论文审计了 1,586 次已完成评估，记录到 9 次正向 Launch Review；描述性汇总中，内容消费渗透率 +5.75%，总消费时长 +10.83%。

---

📄 标题：KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU  
👥 作者：Di Chai, Leye Wang, Zeshen Su, Zhiguo Xia, Zhihang Yu  
🔗 链接：https://arxiv.org/abs/2609.04852  
📝 方法概述：把长运行 agent 的 overflow 历史保存为 paged KV state，并跨 GPU / host memory / NVMe 做检索与调页，只在运行时物化与当前 query 相关的 execution view。  
💡 创新性分析：把“长上下文问题”从 prompt 压缩转移到 workspace virtualization，是一篇非常强的 agent infra 论文。  
📊 关键指标：DeepSWE long-context test 上任务成功率 43.8%→48.4%；本地 24GB RTX 5090 Laptop GPU 可虚拟化到 1M tokens，约 50 tok/s。
