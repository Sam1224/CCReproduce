# EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer |
| 作者 | Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng |
| 机构 | 中科院计算技术研究所 (ICT, CAS) |
| arXiv | https://arxiv.org/abs/2605.01732 |
| 提交日期 | 2026-05-03 |
| 领域标签 | 知识蒸馏 · LLM压缩 · Token级 · 熵引导 · 课程学习 |
| 桶类别 | WEAK |
| 综合评分 | **58 / 100** |

---

## 方法概述 (中文)

LLM 知识蒸馏现有方法对所有 token 一视同仁，忽视了不同 token 对模型决策贡献不均等的现实。高熵 token（模型不确定，需"思考"的关键决策点）被等同于低熵 token（几乎确定的填充词）处理，导致蒸馏效率低下。

**EGAD** 利用教师输出熵在 token 级别动态调整三方面：
1. **Token 级课程**：训练初期聚焦低熵（易）token，随训练进度渐进转向高熵（难）token
2. **自适应损失权重**：根据 token 熵动态缩放蒸馏损失，高熵 token 获得更大权重
3. **分布对齐**：对高熵 token 的概率分布做更细粒度对齐，避免学生在关键决策点偏离教师

---

## 故事线 (Story Arc)

> **现状不足：** MiniLLM、DistiLLM 等蒸馏方法 token 等权处理，在关键推理步骤上学习信号被稀释，模型压缩后关键能力衰退明显。
>
> **我们的解法：** 用教师输出熵作为"难度信号"，让蒸馏的注意力动态集中在真正需要学习的 token 上——EGAD 熵引导自适应蒸馏。

---

## 关键指标

| 任务 | 指标 | EGAD | 基线 (MiniLLM) |
|------|------|------|----------------|
| 推理/QA 基准 | Avg. Accuracy | 改善 ~2-3% | — |
| 指令跟随 | WinRate | 提升 | 基线 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 18 | 30 | 熵引导 token 课程有新意，思路清晰 |
| 实验 SOTA Delta | 8 | 15 | 改进幅度中等 |
| 实验质量/消融 | 9 | 15 | ICT CAS 规范实验 |
| 效率 | 8 | 10 | 蒸馏后模型更小更快 |
| 泛化性 | 3 | 5 | 限于 LLM，未验证 MLLM |
| 领域相关性 | 12 | 25 | 间接：用于压缩内容理解大模型 |
| **总分** | **58** | **100** | — |
