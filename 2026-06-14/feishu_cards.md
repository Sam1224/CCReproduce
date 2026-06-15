# 飞书卡片推送 — 2026-06-14 电商内容生态 & 达人治理巡检

> 本日共发现 **8 篇**得分 ≥ 40 的相关论文（共 9 篇，From AGI to ASI 仅 35 分不纳入推送）。

---

📄 标题：OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model
👥 作者：Xuxin Zhang, Ben Chen, Yue Lv 等（快手科技）
🔗 链接：https://arxiv.org/abs/2606.13533
📝 方法概述：提出 OneRetrieval，通过 Keyword-Aligned Encoding（KAE）将生成式检索的每个 identifier 位置与可解释属性词绑定，并在 codebook 中预留可扩展 slots；运营人员可在部署后将新词显式绑定到 reserved slot，无需 retrain 即可实现"小时级新词注入"，同时通过四阶段 SFT 流水线保证深召回质量。
💡 创新性分析：首次将"可运营编辑性"作为 generative retrieval 的结构性约束而非事后补丁，KAE + reserved slot 把新词路由变成显式绑定动作，兼顾了工业倒排与生成式检索的双重优势。
📊 关键指标：快手搜索日志 Order HR@350=0.5482，Click HR@350=0.6055；线上 A/B Order volume 显著提升，全召回替换后 Item CTR 显著提升。**综合评分：87/100**

---

📄 标题：The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale with Helmsman
👥 作者：Yuchen Huang, Baiteng Ma, Yiping Sun 等（华东师范大学 & 上交 & 小红书）
🔗 链接：https://arxiv.org/abs/2606.13145
📝 方法概述：HELMSMAN 面向十亿级向量检索的生产成本瓶颈，走"全闪存 + 聚类 ANNS"路线，通过三项关键设计补齐性能：用户态 I/O 绕过 kernel 栈以充分利用 NVMe 带宽；基于 top-k 与 query 分布的 learned pruning 自适应裁剪扫描量；GPU 加速建库流水线支撑十亿级 index 的小时级重建。
💡 创新性分析：系统化证明"聚类 + 全闪存"可在严格在线 SLA 下工作，三组件协同解决传统方案的 I/O、裁剪、重建三大痛点，并给出生产规模的真实成本数据。
📊 关键指标：40 台机器承载原需 ~35000 CPU cores + 0.35PB DRAM 的 ANNS 负载，硬件成本节省 >90%；吞吐较 DRAM-SSD ANNS 提升 2–16×，达 in-DRAM ANNS 吞吐约 85%。**综合评分：85/100**

---

📄 标题：ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs
👥 作者：Ashutosh Hathidara, Sai Shruthi Sistla, Sebastian Schreiber, Sahil Bansal（SAP Labs）
🔗 链接：https://arxiv.org/abs/2606.12451
📝 方法概述：为任意工具目录自动生成三类诊断集（更真实的 Realistic Retrieval Benchmark、MCQ Probing、QA Probing），提出 Internalization Score（free-form vs constrained decoding 准确率之差）量化 LLM 对 constrained decoding trie 的依赖程度，揭示"检索能力高但工具知识弱"的解耦现象。
💡 创新性分析：首次把"工具检索能力"与"工具理解能力"解耦测评，并量化 constrained decoding 依赖，直接指导生产型 Agent 上线前的风险审计。
📊 关键指标：部分参数化训练配置在真实 RRB queries 上较标准 ToolBench 评测下滑 50–64 个百分点；部分配置 factual probe 接近随机水平。**综合评分：80/100**

---

📄 标题：Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents
👥 作者：Kushal Raj Bhandari, Ling Yue, Ching-Yun Ko 等（RPI & IBM Research）
🔗 链接：https://arxiv.org/abs/2606.12674
📝 方法概述：把工具调用序列形式化为 typed workflow graph，在推理时通过结构化 edit + 执行反馈 + 自适应强度控制做进化搜索修复，以"执行可行性"为核心目标，避免依赖大规模蒸馏数据或重新训练。
💡 创新性分析：在 schema 与依赖约束下做 workflow repair，比 ReAct/Best-of-N 更贴近真实执行约束；量化展示少量 SFT 数据的潜在负效应，为小模型 tool-use 提供可靠的 test-time 替代方案。
📊 关键指标：MCP-Bench (28 服务器, 250 工具) execution feasibility 从约 3% 提升到 17–24%；相同数据预算下 SFT/DPO 往往降低可用性。**综合评分：80/100**

---

📄 标题：Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants
👥 作者：Shuxian Fan, Seonwoo Min 等（Amazon）
🔗 链接：https://arxiv.org/abs/2606.12608
📝 方法概述：由零售专家编写 525 个 mission（含 293 个多轮），为每个 mission 标注 10,863 条重要性加权二值 rubrics，涵盖偏好澄清/跨商品权衡/兼容性判断等专业能力，并验证 LLM judge 与专家的一致性（macro-F1=0.749）。
💡 创新性分析：首个同时具备专家级 rubrics 与 judge 可靠性验证的多轮电商对话推理评测集，把主观"好建议"拆成可量化的细粒度 criterion，为生产级对话助手提供标准化评测范式。
📊 关键指标：9 个模型 overall weighted pass rate 57.4%–77.2%；多轮退化 4–18 pp；optional vs required rubrics gap 13–29 pp。**综合评分：79/100**

---

📄 标题：CFALR: Collaborative Filtering-Augmented Large Language Model for Personalized Fashion Outfit Recommendation
👥 作者：Yujuan Ding, Junrong Liao, Yunshan Ma 等（香港理工 & UESTC & NUS 等）
🔗 链接：https://arxiv.org/abs/2606.13001
📝 方法概述：把 outfit completion 形式化为 Personalized Fill-In-The-Blank (P-FITB) 生成任务，将 CF 向量与多模态特征投影为单 token 注入 LLM，推理时通过 CF-augmented 生成机制融合协同信号与语言模型分布，兼顾个性化与语义理解。
💡 创新性分析：首次在 outfit-level 组合推荐中实现显式 CF+LLM 特征 token 融合，P-FITB 任务化把组合推荐变成可监督的生成式学习，稀疏设置下 CF 与视觉特征互补性显著。
📊 关键指标：Polyvore 数据集 P-FITB Accuracy：CFALR 0.6498（1/4）vs Bundle-MLLM 0.4775；IQON 数据集同样大幅领先。**综合评分：78/100**

---

📄 标题：GRIP: Feedback-Guided Prompt Retrieval for Large Multimodal Models
👥 作者：Garvita Allabadi, Matteo Sodano 等（UIUC & Univ. Bonn & Microsoft）
🔗 链接：https://arxiv.org/abs/2606.12744
📝 方法概述：将多模态 ICL 的示例检索监督改为 LMM 实际反馈：对候选邻居逐一做加入/不加入对比实验，按任务指标增益标注 beneficial/detrimental，再用对比学习训练 vision-only retriever，使检索空间对齐"带来性能提升"的 utility，并可迁移到闭源模型。
💡 创新性分析：首次把 utility alignment 扩展到多模态 ICL 检索，vision-only retriever 可跨 LMM（含 GPT-4o）迁移，工程价值高；适用于多模态 RAG 的 few-shot 示例优化。
📊 关键指标：Qwen2.5-VL-7B 上 Oxford Pets 83.9% vs ViT 79.0%；GPT-4o ScienceVQA 85.9% vs zero-shot 79.3%。**综合评分：72/100**

---

📄 标题：TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering
👥 作者：Rawan Hesham, Ali Ashraf 等（Capital University Egypt）
🔗 链接：https://arxiv.org/abs/2606.13267
📝 方法概述：端侧（YOLOv8n, 5.97MB）识别 + 后端 RAG（ChromaDB + 量化 Gemma 4）的分层双语导览系统，核心经验在"数据治理优先于模型规模"——先用 YOLO-World 自动标注，再用空间清洗规则修 label，最后人工闭环，端到端延迟从 >30s 优化至约 10s。
💡 创新性分析：端侧识别+RAG QA 的分层架构 + foundation-model auto-annotation 清洗流水线，证明在细粒度视觉类别里 label 质量决定成败，为电商商品识别+知识问答场景提供完整工程范式。
📊 关键指标：识别 mAP@0.5=0.995，mAP@0.5:0.95=0.924，模型 5.97MB，端到端延迟约 10s。**综合评分：65/100**
