# Feishu Cards — 2026-05-10 Daily AI Paper Inspection
# 飞书卡片 — 2026-05-10 日报 (分数 ≥ 40 的论文)

---

📄 标题：VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform
👥 作者：Xingyu Lu, Tianke Zhang, Chang Meng et al. (快手 Kuaishou Technology)
🔗 链接：https://arxiv.org/abs/2504.14904
📝 方法概述：提出 KuaiMod，将视觉语言大模型（VLM）训练为基于"普通法判例推理"的动态内容审核策略执行者。通过从真实平台抽取违规/合规案例构建 CoT 训练数据，结合课程训练与在线反馈持续精化策略。首创短视频平台（SVP）内容审核基准，包含真实用户/审核员反馈。
💡 创新性分析：首次将"普通法（Common-Law）"类比引入内容审核，VLM 作为活策略执行者而非静态分类器；在线反馈闭环支持策略动态演化；与传统规则/分类器相比，推理可解释性大幅提升。评分：83/100
📊 关键指标：生产部署后用户举报率下降 20%；在多个快手推荐场景 DAU 和 APP 使用时长均显著提升（KDD 2026 收录）

---

📄 标题：Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching
👥 作者：Wei Chee Yew, Hailun Xu, Sanjay Saha, Xiaotian Fan et al.
🔗 链接：https://arxiv.org/abs/2512.03553
📝 方法概述：针对直播电商/游戏/互动场景，设计双路径混合审核系统：监督分类路径检测已知违规，MLLM 增强相似度匹配路径处理新型/模糊违规，MLLM 知识蒸馏保证实时性。文本/音频/视觉多模态同步处理，两路融合取高置信度裁决。
💡 创新性分析：双路互补架构兼顾精度与泛化性；MLLM 蒸馏将大模型推理能力迁移至轻量路由器；专为直播时序特性设计，不同于离线审核范式。评分：79/100
📊 关键指标：生产平台（百万小时内容）分类路径 Recall@80%P=67%，相似度路径 Recall@80%P=76%，远超单路系统。KDD 2026 收录。

---

📄 标题：AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce
👥 作者：Biao Zhang, Lixin Chen, Bin Zhang, Zongwei Wang, Tong Liu, Bo Zheng（阿里巴巴淘宝天猫集团）
🔗 链接：https://arxiv.org/abs/2604.20135
📝 方法概述：针对电商同款商品检索中的细粒度语义辨别挑战，提出 AFMRL：先用 MLLM 自动提取商品属性（颜色/材质/款式），再通过属性引导对比学习（AGCL）做难负例挖掘，最后用检索感知属性强化（RAR）做 RL 精化，三阶段闭环显著提升细粒度表示质量。
💡 创新性分析：将生成式 MLLM 与判别式对比学习+强化学习结合，属性作为中间语义锚点突破细粒度识别瓶颈；阿里巴巴大规模电商数据验证，工业可信度高。评分：78/100
📊 关键指标：在大规模电商检索数据集上多任务 SOTA，超越 VLM2Vec 等通用大模型；多类目同款检索任务全面领先。

---

📄 标题：Adapting Vision-Language Models for E-commerce Understanding at Scale
👥 作者：Matteo Nulli, Vladimir Orshulevich et al. (12 authors)
🔗 链接：https://arxiv.org/abs/2602.11733
📝 方法概述：系统研究如何将通用 VLM 适配至电商多图像/属性中心/噪声数据场景。构建覆盖深度商品理解、严格指令跟随、动态属性提取三维度的新评估套件，通过大规模消融实验提炼最佳实践方法论。
💡 创新性分析：填补电商 VLM 适配方法论空白，为行业提供系统性参考；三维评估套件标准化电商 VLM 评测。评分：74/100
📊 关键指标：有针对性适配后，电商多维评估指标全面显著提升，同时保持通用能力；评估套件计划开源。

---

📄 标题：OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents
👥 作者：Shuang Chen et al. (10 authors)
🔗 链接：https://arxiv.org/abs/2605.05185 | GitHub: https://github.com/shawn0728/OpenSearch-VL
📝 方法概述：开源完整多模态深度搜索 Agent 训练方案，包含：①Wikipedia 路径采样+模糊改写的高质数据策划（SFT-36k/RL-8k）；②整合文本/图像/OCR/超分等 7 类工具的统一环境；③Fatal-Aware GRPO 算法（工具失败后 token mask + 单边优势截断）防止致命失败污染奖励。
💡 创新性分析：首个完整开放的多模态搜索 Agent 配方；Fatal-Aware GRPO 优雅处理工具失败，工程创新明显；7 项基准平均+10分，可商业模型比肩。评分：76/100
📊 关键指标：7项多模态搜索基准平均提升 10+ 分；BrowseComp-Plus 与商业闭源模型持平；代码/数据/模型全部开源。

---

📄 标题：Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe
👥 作者：Wenjin Hou et al. (16 authors)（清华大学 THUNLP 等）
🔗 链接：https://arxiv.org/abs/2605.03677 | GitHub: https://github.com/thunlp/OPD
📝 方法概述：提出统一 On-Policy 蒸馏框架 Uni-OPD，从学生视角引入双平衡数据策略，从教师视角提出 Outcome-Guided Margin Calibration（OGMC）恢复教师监督序一致性。统一覆盖 LLM 和 MLLM，支持单教师/多教师/跨模态蒸馏。
💡 创新性分析：首个统一 LLM/MLLM 的 OPD 框架；OGMC 校准教师可靠性是关键创新；5 领域 16 基准全面验证，清华强实验室。评分：72/100
📊 关键指标：5 领域（推理/QA/指令跟随/多模态/跨模态）16 基准全面超越 DistiLLM、MiniLLM 等基线；跨模态蒸馏首次系统验证。

---

📄 标题：Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation
👥 作者：Siyuan Li, Aodu Wulianghai, Xi Lin, Xibin Yuan, Qinghua Mao, Guangyan Li, Xiang Chen, Jun Wu, Jianhua Li
🔗 链接：https://arxiv.org/abs/2605.05950
📝 方法概述：提出 LiSCP，通过构建跨改写变体的文体一致性侧写来检测 LLM 生成文本：结合离散文体特征（句式/标点/词性）和连续语义信号（embedding 相似度），轻量分类头基于一致性剖面做判断，对对抗性改写具有强鲁棒性。
💡 创新性分析：改变检测维度——从"表面文字"转向"风格一致性指纹"，在对抗改写场景下 DetectGPT 等现有方法退化到接近随机，LiSCP 保持稳定；轻量设计适合大规模内容平台。评分：67/100
📊 关键指标：对抗改写测试集 AUROC 显著高于 DetectGPT（后者约 55%，接近随机）；轻量级，适合实时内容审核部署。

---

📄 标题：Multimodal Data Curation Through Ranked Retrieval
👥 作者：Pratyush Muthukumar, Harshil Kotamreddy et al.（NVIDIA）
🔗 链接：https://arxiv.org/abs/2605.01163
📝 方法概述：针对多模态 embedding 空间模态驱动分离问题，提出 SNS（Symmetric Nucleus Subsampling，对称核子采样）修剪噪声训练对，以及 EEE（Expert Embedding Engine，多专家 embedding 融合）减少模态偏差，整体提升跨模态检索和数据策划质量。
💡 创新性分析：从训练对和 embedding 模型两个层面同时修复模态偏差；SNS 无监督过滤噪声配对，EEE 学习投影层融合专家。评分：63/100
📊 关键指标：跨模态检索 R@1 显著改善；下游数据策划任务训练效率提升；NVIDIA 实验规范。

---

📄 标题：EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer
👥 作者：Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng（中科院计算技术研究所）
🔗 链接：https://arxiv.org/abs/2605.01732
📝 方法概述：利用教师输出熵在 token 级别动态调整蒸馏：高熵 token（关键决策点）获得更大训练权重和更细粒度分布对齐，采用由易到难的 token 级课程学习策略，解决现有蒸馏方法 token 等权处理的低效问题。
💡 创新性分析：以教师熵作为 token 难度信号驱动蒸馏课程，视角新颖；ICT CAS 实验。评分：58/100
📊 关键指标：推理/QA 基准平均准确率提升 2-3%；指令跟随 WinRate 优于 MiniLLM 等基线。
