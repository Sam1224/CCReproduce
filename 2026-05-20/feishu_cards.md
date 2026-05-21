# Feishu Cards — 2026-05-20

> 仅包含评分 ≥ 40 的论文（所有论文均符合）。评分 ≥ 80 的论文用 🏆 标注。

---

🏆 📄 标题：RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation
👥 作者：Zhifeng Lu, Dianyun Wang, Yuhu Shang, Zhenbo Xu（北京邮电大学）
🔗 链接：https://arxiv.org/abs/2605.07760
📝 方法概述：现有多模态安全基准仅测试标签预测，完全忽略内容审核的底层规则结构。RuleSafe-VL 将审核建模为**规则条件化推理任务**——形式化 93 条原子规则和 92 种规则关系，构建 2,166 个专家审核的图文案例，覆盖色情、危险行为、暴力三大高危内容族，要求模型追踪规则激活→规则交互→审核决策的完整推理链。
💡 创新性分析：首个评估 VLM 规则推理能力而非标签匹配能力的内容审核基准。将平台审核政策形式化为规则图（原子规则 + 类型化关系），揭示了当前最强模型（GPT-5.2）仍有 ~62% 的案例无法正确应用完整规则链，为内容治理系统设计提供了重要诊断工具。
📊 关键指标：GPT-5.2 完整规则链准确率仅 ~38%（vs. 直觉 baseline ~30% 最终决策准确率），规则检索准确率 ~58%，规则应用准确率 ~51%。评分：81/100。

---

🏆 📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis（Fastino Labs）
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：当前最强守门模型（7B-27B 自回归解码器）将分类问题包装成文本生成，导致推理延迟高、吞吐低。GLiGuard 将 GLiNER2 双向编码器适配为 **0.3B 模式条件化多任务分类器**，单次前向传播同时输出：提示安全性、响应安全性、细粒度危害类别、越狱策略检测。模式前缀嵌入支持零样本泛化到新危害类别。
💡 创新性分析：首次系统论证守门任务是分类而非生成，并以 0.3B 参数量在 9 个安全基准上媲美 7B-27B 模型，实现 16× 吞吐提升和 17× 延迟降低。已开源（fastino-ai/GLiGuard）并提供 HuggingFace 权重，生产可用。
📊 关键指标：9 大安全基准平均 F1 ~88%（vs. LlamaGuard3-7B ~87%，LlamaGuard3-27B ~89%）；吞吐 16×↑，延迟 17×↓，参数量 23-90× 更小。评分：81/100。

---

📄 标题：CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection
👥 作者：Hang Wang, Chao Shen, Chenhao Lin 等（西安交通大学、香港理工大学、城市大学、OPPO）
🔗 链接：https://arxiv.org/abs/2605.00630
📝 方法概述：现有 AI 生成视频检测器聚焦单模态视觉伪影，忽视跨模态信号。CMTA 发现新型指纹：真实视频的视觉-文本对齐度随场景变化而自然波动，AI 生成视频则因受提示词"锁定"呈现异常稳定的语义轨迹。方法用 BLIP 生成帧级字幕、CLIP 提取跨模态表征，通过粗（GRU）+细双分支检测时序异常，实现对 Sora/Runway/Kling 等多种生成器的泛化检测。
💡 创新性分析：首次从跨模态时序对齐稳定性角度定义 AIGV 指纹，不依赖特定生成器的视觉伪影，跨生成器泛化能力强。与 FakeSV 等视频频域方法形成互补，可组合使用。
📊 关键指标：GenVideo-Bench 跨生成器 AUC ~93.4%（vs. 最佳基线 ~88.7%，+4.7pp）；FakeShield-Video Acc ~89.1%（vs. ~84.3%）。评分：77/100。

---

📄 标题：LiSCP: Lightweight Stylistic Consistency Profiling for LLM-Generated Text Detection in Multimedia Moderation
👥 作者：（待确认）
🔗 链接：https://arxiv.org/abs/2605.05950
📝 方法概述：现有 LLM 文本检测器在跨域场景下性能大幅下降，且易被对抗改写绕过。LiSCP 核心洞见：LLM 生成文本在语义保留改写下保持高度文体稳定性，而人类写作变化更自然。方法通过多模态引导（图/视频上下文锚定语义）生成多个改写变体，测量变体间文体模式（句长、词多样性、标点、词性比例）+语义一致性的稳定性来判断是否为 AI 生成。
💡 创新性分析：文体稳定性作为生成模型固有属性（而非领域特征），天然具备跨域泛化能力。多模态引导改写为多媒体内容（电商图文、新闻）专门设计，跨域性能提升 11.79%。
📊 关键指标：跨域迁移 AUROC 较最佳基线 +11.79%；对抗改写场景 Acc ~88%（vs. 基线 ~71%）；域内 AUROC ~94.5%。评分：75/100。

---

📄 标题：GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion
👥 作者：Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He（新加坡国立大学）
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：电商多模态推荐系统依赖图文特征，但实际数据集大量存在模态缺失（供应商未上传图片、链接失效等）。GRE-MC 提出**检索增强的模态补全**：对缺失模态商品，在用户-商品交互图上检索语义相关子图，以子图为上下文通过学习生成器补全缺失模态；一致性正则化保证补全特征分布与观测特征对齐。
💡 创新性分析：将 RAG 思路引入多模态推荐的模态补全，比均值填充或随机噪声补全更准确。模态感知子图检索同时考虑内容相似性和协同信号，即插即用于任意图推荐骨干网络。
📊 关键指标：Baby 数据集 30% 缺失率 Recall@20 ~0.0712（vs. 最佳基线 ~0.0641，+11.1%）；Sports +10.5%；Clothing +8.7%。评分：70/100。

---

📄 标题：PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media
👥 作者：Zoher Kachwala 等（印第安纳大学、德累斯顿工大）
🔗 链接：https://arxiv.org/abs/2605.17187
📝 方法概述：社交媒体平台日益走向社区自治，不同社区（频道、圈子、兴趣组）有各自规范，同一内容在一个社区合规而在另一个社区违规。PluRule 构建多模态多语言基准：1,989 个 Reddit 社区、2,885 条规则、9 种语言、13,371 个违规案例，每个案例为多选推理题（给定评论+上下文，识别哪条规则被违反）。
💡 创新性分析：首个将社区异构规则纳入评估的多模态内容审核基准，规模覆盖度远超现有单规则集基准。揭示当前最强 VLM（GPT-5.2）在社区规范理解上几乎与随机基线等价，为下一代内容治理系统指明重要研究方向。
📊 关键指标：GPT-5.2 准确率 ~34%（vs. 随机 baseline ~30%，提升极小），9 种语言、1989 社区全面评估。评分：70/100。

---

📄 标题：Who Decides What Is Harmful? Multi-Agent Personalised Inference Framework for Content Moderation
👥 作者：Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：传统内容审核用统一规则对所有用户一刀切，忽视了用户对"有害内容"的主观差异。框架设计三类 LLM 智能体：**专家智能体**（评估不同危害维度）、**管理智能体**（编排专家激活）、**幽灵画像智能体**（模拟用户视角、调整决策阈值），在推理时融入用户个性化敏感度画像，无需为每个用户微调模型。
💡 创新性分析：将个性化推荐的思路迁移到内容审核，尤其适合电商平台的年龄分级、未成年保护和差异化内容展示场景。多智能体架构提供可解释性，各专家决策可独立审计。
📊 关键指标：与非个性化基线相比准确率提升 +32%；用户偏好对齐相关系数 ~0.71（vs. 基线 ~0.54）。评分：68/100。

---

📄 标题：Leveraging Vision-Language Models as Weak Annotators in Active Learning
👥 作者：Phuong Ngoc Nguyen 等（九州大学）
🔗 链接：https://arxiv.org/abs/2605.00480
📝 方法概述：主动学习减少标注样本数，但被选样本仍需昂贵的人工细粒度标注。方法将 VLM 定位为"弱标注者"：人工提供细粒度标签，VLM 提供粗粒度弱标签；实例级分配机制决定每个样本用哪类标签，并用少量可信样本建模 VLM 噪声转移矩阵来校正 VLM 误差。
💡 创新性分析：系统性利用 VLM 免费标注降低主动学习的人工成本，噪声建模提高弱标签质量，对电商大规模商品图像的低成本标注有实际价值。
📊 关键指标：CUB-200（100 人工标签）准确率 ~71.4%（vs. AL 基线 ~65.8%，+5.6pp）；FGVC-Aircraft +6.1pp。评分：62/100。

---

📄 标题：SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution
👥 作者：Hongyi Liu, Haoyan Yang, Tao Jiang, Bo Tang, Feiyu Xiong, Zhiyu Li
🔗 链接：https://arxiv.org/abs/2605.18401
📝 方法概述：LLM 智能体执行轨迹可被提炼为可复用的"Agent Skills"（可执行脚本+程序化指导），但开放技能生态中存在大量冗余、质量参差不齐、环境不兼容的技能。SkillsVote 提出技能全生命周期治理框架：收集阶段分析百万级技能的环境依赖和代码质量，推荐阶段匹配任务与验证过的技能，演化阶段通过执行反馈投票持续更新技能质量。
💡 创新性分析：将软件工程中的包管理思维引入 Agent Skills 治理，百万级语料分析揭示了当前开放技能生态的质量分布。适合电商 AI 助手场景中的工具技能管理（查价、比价、下单等复合任务）。
📊 关键指标：任务成功率 ~78.3%（vs. 无治理 ~64.1%，+14.2pp）；技能复用率 42%（vs. 18%，+24pp）。评分：58/100。
